"""Contract for turning model prose extraction into structured, auditable claims.

The rule based attempt at this (`recall/fix.py`) proposed ZERO edges on a real 792 memo
corpus: four candidates survived its mechanical rules and all four were wrong on review.
So every property here is a REFUSAL property. The properties, one test each:

Validation ladder, in the order `_normalize` applies it:

1. Output that is not JSON rejects the whole file's output.
2. Output whose top level shape is wrong rejects the whole file's output.
3. More than `MAX_CLAIMS_PER_FILE` claims rejects the whole file's output.
4. An unknown `kind`, a wrong field type, an off vocabulary status, or a malformed date
   rejects the whole batch — the model ignored the contract, so none of it is trusted.
5. A `quote` that is not a verbatim substring of `human_body` rejects THAT claim. This is
   the strongest guard in the design: it is what makes a claim checkable by a human.
6. A supersession target that is not in `corpus_names` after `supersedes_key` normalisation
   rejects THAT claim, and an ambiguous target is refused rather than guessed at — the same
   refusal `recall/fix.py` makes.
7. A date not literally present in `human_body` rejects THAT claim.

Ladder ORDER is itself a property: a claim failing several rungs is reported against the
earliest one, so a reviewer sees the strongest reason.

Beyond the ladder:

- The quote is checked against the body with frontmatter REMOVED, so a model cannot "quote"
  the very frontmatter the extraction is supposed to justify writing.
- Surviving claims are canonicalised: a supersession target resolves to the corpus name.
- Proposal ids are stable content hashes, and `_providers._coerce_provider_proposal`
  recomputes them, so the adapter cannot invent an identity.
- DIRECTION: `subject_id` is the SUPERSEDED document and `object_id` the superseding one.
  Getting this backwards declares the live memo stale and demotes it beneath the one it
  replaced, which is the exact failure the trust layer exists to prevent. Pinned against an
  asymmetric fixture where the two documents are not interchangeable.
- The deterministic reference reproduces at least the supersessions `_deterministic.py`
  already finds, so replacing rules with a model cannot silently lose recall.
- The cache answers on identical input and is keyed on engine identity and prompt revision.
- Extraction is an ingest path concern: the query planner never imports it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from recall.reasoning_graph import ReasoningGraphProjection, build_reasoning_graph
from recall.reasoning_proposals import (
    PROPOSAL_SCHEMA_VERSION,
    deterministic_inference_proposals,
    proposal_report,
)
from recall.truth_extraction import (
    ExtractedClaimProposalProvider,
    CLAIM_KINDS,
    DETERMINISTIC_EXTRACTION_ENGINE_ID,
    MAX_CLAIMS_PER_FILE,
    PROMPT_REVISION,
    STATUS_VOCABULARY,
    DeterministicExtractionEngine,
    ExtractionBatchRejected,
    IdentityClaim,
    StatusClaim,
    SupersessionClaim,
    ValidityClaim,
    build_extraction_prompt,
    human_body_of,
    normalize_extraction,
    resolve_extraction_engine,
)
from recall.truth_extraction._cache import (
    InMemoryExtractionCache,
    extraction_cache_key,
)
from recall.truth_extraction.extract import extract_corpus_claims, extract_file_claims
from recall.types import Chunk
from tests.fakes import FakeExtractionEngine

BODY = (
    "This memo supersedes archive_policy_2026-01-05.md after the January review.\n"
    "Effective valid_from 2026-02-01 the retention window shortens to thirty days.\n"
    "Status: deprecated as of the February rollout.\n"
    "The Retention Board (formerly the Archive Council) signed off.\n"
)
CORPUS = ("archive_policy_2026-01-05.md", "retention_policy_2026-02-01.md")
FILE = "retention_policy_2026-02-01.md"

SUPERSESSION = {
    "kind": "supersession",
    "superseded": "archive_policy_2026-01-05",
    "quote": "This memo supersedes archive_policy_2026-01-05.md after the January review.",
}
VALIDITY = {
    "kind": "validity",
    "key": "valid_from",
    "date": "2026-02-01",
    "quote": "Effective valid_from 2026-02-01 the retention window shortens to thirty days.",
}
STATUS = {
    "kind": "status",
    "value": "deprecated",
    "quote": "Status: deprecated as of the February rollout.",
}
IDENTITY = {
    "kind": "identity",
    "entity": "Retention Board",
    "alias": "Archive Council",
    "quote": "The Retention Board (formerly the Archive Council) signed off.",
}


def _payload(*claims: dict) -> str:
    return json.dumps({"claims": list(claims)})


def _normalize(raw: str, *, human_body: str = BODY, corpus_names: tuple[str, ...] = CORPUS):
    return normalize_extraction(raw, file=FILE, human_body=human_body, corpus_names=corpus_names)


def test_every_claim_kind_survives_a_well_formed_batch() -> None:
    accepted, rejected = _normalize(_payload(SUPERSESSION, VALIDITY, STATUS, IDENTITY))

    assert rejected == ()
    assert [claim.kind for claim in accepted] == ["supersession", "validity", "status", "identity"]
    assert isinstance(accepted[0], SupersessionClaim)
    assert isinstance(accepted[1], ValidityClaim)
    assert isinstance(accepted[2], StatusClaim)
    assert isinstance(accepted[3], IdentityClaim)


# --- rung 1: JSON parse -------------------------------------------------------------------


def test_output_that_is_not_json_rejects_the_whole_file() -> None:
    with pytest.raises(ExtractionBatchRejected, match="json") as excinfo:
        _normalize("I found one supersession claim in this memo.")

    assert excinfo.value.rung == "json"


# --- rung 2: top level shape --------------------------------------------------------------


def test_top_level_array_rejects_the_whole_file() -> None:
    with pytest.raises(ExtractionBatchRejected, match="top_level_shape") as excinfo:
        _normalize(json.dumps([SUPERSESSION]))

    assert excinfo.value.rung == "top_level_shape"


def test_top_level_claims_that_is_not_a_list_rejects_the_whole_file() -> None:
    with pytest.raises(ExtractionBatchRejected, match="top_level_shape"):
        _normalize(json.dumps({"claims": SUPERSESSION}))


# --- rung 3: cardinality ------------------------------------------------------------------


def test_more_claims_than_the_per_file_maximum_rejects_the_whole_file() -> None:
    with pytest.raises(ExtractionBatchRejected, match="max_claims") as excinfo:
        _normalize(_payload(*([SUPERSESSION] * (MAX_CLAIMS_PER_FILE + 1))))

    assert excinfo.value.rung == "max_claims"


def test_exactly_the_per_file_maximum_is_accepted() -> None:
    accepted, rejected = _normalize(_payload(*([SUPERSESSION] * MAX_CLAIMS_PER_FILE)))

    assert len(accepted) == MAX_CLAIMS_PER_FILE
    assert rejected == ()


# --- rung 4: claim shape ------------------------------------------------------------------


def test_unknown_kind_rejects_the_whole_batch() -> None:
    with pytest.raises(ExtractionBatchRejected, match="claim_shape") as excinfo:
        _normalize(_payload(SUPERSESSION, {"kind": "sentiment", "quote": "x"}))

    assert excinfo.value.rung == "claim_shape"


def test_wrong_field_type_rejects_the_whole_batch() -> None:
    with pytest.raises(ExtractionBatchRejected, match="claim_shape"):
        _normalize(_payload({**SUPERSESSION, "superseded": ["archive_policy_2026-01-05"]}))


def test_missing_field_rejects_the_whole_batch() -> None:
    with pytest.raises(ExtractionBatchRejected, match="claim_shape"):
        _normalize(_payload({"kind": "supersession", "quote": SUPERSESSION["quote"]}))


def test_off_vocabulary_status_rejects_the_whole_batch() -> None:
    with pytest.raises(ExtractionBatchRejected, match="claim_shape"):
        _normalize(_payload({**STATUS, "value": "vibes"}))


def test_malformed_date_rejects_the_whole_batch() -> None:
    with pytest.raises(ExtractionBatchRejected, match="claim_shape"):
        _normalize(_payload({**VALIDITY, "date": "the first of February"}))


def test_unknown_validity_key_rejects_the_whole_batch() -> None:
    with pytest.raises(ExtractionBatchRejected, match="claim_shape"):
        _normalize(_payload({**VALIDITY, "key": "expires"}))


def test_status_vocabulary_is_closed() -> None:
    for value in STATUS_VOCABULARY:
        accepted, _rejected = _normalize(
            _payload({**STATUS, "value": value, "quote": "Status: deprecated"})
        )
        assert accepted[0].value == value


# --- rung 5: quote must be verbatim -------------------------------------------------------


def test_quote_that_is_not_a_substring_of_the_body_rejects_that_claim() -> None:
    paraphrased = {**SUPERSESSION, "quote": "This memo replaces the January archive policy."}

    accepted, rejected = _normalize(_payload(SUPERSESSION, paraphrased))

    assert len(accepted) == 1
    assert [rejection.rung for rejection in rejected] == ["quote_not_verbatim"]
    assert rejected[0].index == 1


def test_quote_from_the_frontmatter_block_is_not_verbatim_body() -> None:
    document = "---\nsupersedes: archive_policy_2026-01-05.md\n---\n" + BODY
    body = human_body_of(document)
    forged = {**SUPERSESSION, "quote": "supersedes: archive_policy_2026-01-05.md"}

    accepted, rejected = _normalize(_payload(forged), human_body=body)

    assert accepted == ()
    assert rejected[0].rung == "quote_not_verbatim"


def test_human_body_of_strips_the_frontmatter_block() -> None:
    document = "---\nvalid_from: 2026-02-01\n---\n" + BODY

    assert human_body_of(document) == BODY
    assert "valid_from: 2026-02-01" not in human_body_of(document)


# --- rung 6: target must be in the corpus -------------------------------------------------


def test_supersession_target_outside_the_corpus_rejects_that_claim() -> None:
    outside = {
        **SUPERSESSION,
        "superseded": "policy_that_does_not_exist",
        "quote": "Status: deprecated as of the February rollout.",
    }

    accepted, rejected = _normalize(_payload(SUPERSESSION, outside))

    assert len(accepted) == 1
    assert [rejection.rung for rejection in rejected] == ["target_not_in_corpus"]


def test_ambiguous_supersession_target_is_refused_not_guessed() -> None:
    ambiguous_corpus = (*CORPUS, "archived/archive_policy_2026-01-05.md")

    accepted, rejected = _normalize(_payload(SUPERSESSION), corpus_names=ambiguous_corpus)

    assert accepted == ()
    assert rejected[0].rung == "target_not_in_corpus"
    assert "2 files" in rejected[0].reason


def test_supersession_target_resolves_through_supersedes_key_normalisation() -> None:
    wikilinked = {**SUPERSESSION, "superseded": "[[archive_policy_2026-01-05]]"}

    accepted, rejected = _normalize(_payload(wikilinked))

    assert rejected == ()
    assert accepted[0].superseded == "archive_policy_2026-01-05.md"


def test_self_supersession_is_refused() -> None:
    itself = {**SUPERSESSION, "superseded": FILE}

    accepted, rejected = _normalize(_payload(itself))

    assert accepted == ()
    assert rejected[0].rung == "target_not_in_corpus"


# --- rung 7: date must be present in the body ---------------------------------------------


def test_date_absent_from_the_body_rejects_that_claim() -> None:
    invented = {**VALIDITY, "date": "2026-03-15"}

    accepted, rejected = _normalize(_payload(VALIDITY, invented))

    assert len(accepted) == 1
    assert [rejection.rung for rejection in rejected] == ["date_not_in_body"]


# --- ladder ORDER -------------------------------------------------------------------------


def test_a_claim_failing_several_rungs_is_reported_against_the_earliest() -> None:
    both = {
        "kind": "supersession",
        "superseded": "policy_that_does_not_exist",
        "quote": "a quote the author never wrote",
    }

    _accepted, rejected = _normalize(_payload(both))

    assert rejected[0].rung == "quote_not_verbatim"


def test_a_validity_claim_with_a_bad_quote_is_reported_against_the_quote_rung() -> None:
    both = {**VALIDITY, "date": "2026-03-15", "quote": "a quote the author never wrote"}

    _accepted, rejected = _normalize(_payload(both))

    assert rejected[0].rung == "quote_not_verbatim"


# --- the prompt is pure ---------------------------------------------------------------------


def test_prompt_states_every_claim_kind_and_the_closed_status_vocabulary() -> None:
    prompt = build_extraction_prompt(file=FILE, human_body=BODY, corpus_names=CORPUS)

    for kind in CLAIM_KINDS:
        assert kind in prompt.user
    for value in STATUS_VOCABULARY:
        assert value in prompt.user


def test_prompt_states_the_verbatim_quote_rule_and_the_per_file_maximum() -> None:
    prompt = build_extraction_prompt(file=FILE, human_body=BODY, corpus_names=CORPUS)

    assert "verbatim" in prompt.user.lower()
    assert str(MAX_CLAIMS_PER_FILE) in prompt.user


def test_prompt_carries_the_body_and_the_corpus_names_the_ladder_will_check_against() -> None:
    # A name that appears NOWHERE else in the prompt. Both members of CORPUS do: one is the
    # document under extraction and the other is named in its body, so asserting on them
    # would pass even if the corpus listing were dropped entirely.
    unmentioned = "unmentioned_policy_2025-11-02.md"
    prompt = build_extraction_prompt(
        file=FILE, human_body=BODY, corpus_names=(*CORPUS, unmentioned)
    )

    assert BODY in prompt.user
    assert unmentioned in prompt.user


def test_prompt_is_a_pure_function_of_its_inputs() -> None:
    first = build_extraction_prompt(file=FILE, human_body=BODY, corpus_names=CORPUS)
    second = build_extraction_prompt(file=FILE, human_body=BODY, corpus_names=tuple(CORPUS))

    assert first == second
    assert first.revision == PROMPT_REVISION


# --- the deterministic reference engine ------------------------------------------------------


def _deterministic_claims(body: str, *, file: str = FILE, corpus=CORPUS):
    engine = DeterministicExtractionEngine()
    prompt = build_extraction_prompt(file=file, human_body=body, corpus_names=corpus)
    return normalize_extraction(
        engine.run(prompt), file=file, human_body=body, corpus_names=corpus
    )


def test_deterministic_engine_output_survives_its_own_ladder() -> None:
    accepted, rejected = _deterministic_claims(BODY)

    assert rejected == ()
    assert {claim.kind for claim in accepted} == {
        "supersession",
        "validity",
        "status",
        "identity",
    }


def test_deterministic_engine_quotes_are_verbatim_body_substrings() -> None:
    accepted, _rejected = _deterministic_claims(BODY)

    assert accepted
    for claim in accepted:
        assert claim.quote in BODY


def test_deterministic_engine_emits_nothing_for_a_body_with_no_markers() -> None:
    accepted, rejected = _deterministic_claims("A memo about nothing in particular.\n")

    assert accepted == ()
    assert rejected == ()


def test_deterministic_engine_refuses_passive_voice_rather_than_inverting_it() -> None:
    passive = "This memo is superseded by archive_policy_2026-01-05.md.\n"

    accepted, rejected = _deterministic_claims(passive)

    assert [claim for claim in accepted if claim.kind == "supersession"] == []
    assert rejected == ()


# --- engine resolution -----------------------------------------------------------------------


def test_extraction_is_off_unless_explicitly_enabled() -> None:
    assert resolve_extraction_engine({}) is None
    assert resolve_extraction_engine({"RECALL_TRUTH_EXTRACTION": "0"}) is None


def test_non_boolean_extraction_flag_is_refused() -> None:
    with pytest.raises(ValueError, match="RECALL_TRUTH_EXTRACTION"):
        resolve_extraction_engine({"RECALL_TRUTH_EXTRACTION": "maybe"})


def test_enabling_extraction_yields_the_deterministic_engine_by_default() -> None:
    engine = resolve_extraction_engine({"RECALL_TRUTH_EXTRACTION": "1"})

    assert isinstance(engine, DeterministicExtractionEngine)
    assert engine.engine_id == DETERMINISTIC_EXTRACTION_ENGINE_ID


def test_unknown_engine_name_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="RECALL_TRUTH_EXTRACTION_ENGINE"):
        resolve_extraction_engine(
            {"RECALL_TRUTH_EXTRACTION": "1", "RECALL_TRUTH_EXTRACTION_ENGINE": "gpt"}
        )


# --- cache and orchestration -----------------------------------------------------------------

DOCUMENT = "---\nvalid_from: 2026-02-01\n---\n" + BODY
DOCUMENTS = {FILE: DOCUMENT, "archive_policy_2026-01-05.md": "The original archive policy.\n"}


def _fake(raw: str | None = None) -> FakeExtractionEngine:
    return FakeExtractionEngine({FILE: raw} if raw is not None else None)


def test_extraction_quotes_are_checked_against_the_body_not_the_frontmatter() -> None:
    engine = _fake(_payload({**SUPERSESSION, "quote": "valid_from: 2026-02-01"}))

    result = extract_file_claims(file=FILE, text=DOCUMENT, corpus_names=CORPUS, engine=engine)

    assert result.claims == ()
    assert result.rejections[0].rung == "quote_not_verbatim"


def test_a_batch_rejection_is_recorded_for_review_rather_than_raised() -> None:
    engine = _fake("not json at all")

    result = extract_file_claims(file=FILE, text=DOCUMENT, corpus_names=CORPUS, engine=engine)

    assert result.claims == ()
    assert result.batch_rejection is not None
    assert result.batch_rejection.rung == "json"
    assert result.batch_rejection.index == -1


def test_extraction_records_the_engine_identity_that_produced_it() -> None:
    engine = _fake(_payload(SUPERSESSION))

    result = extract_file_claims(file=FILE, text=DOCUMENT, corpus_names=CORPUS, engine=engine)

    assert (result.engine_id, result.model_id, result.revision) == (
        engine.engine_id,
        engine.model_id,
        engine.revision,
    )


def test_the_cache_answers_the_second_identical_extraction() -> None:
    engine = _fake(_payload(SUPERSESSION))
    cache = InMemoryExtractionCache()

    first = extract_file_claims(
        file=FILE, text=DOCUMENT, corpus_names=CORPUS, engine=engine, cache=cache
    )
    second = extract_file_claims(
        file=FILE, text=DOCUMENT, corpus_names=CORPUS, engine=engine, cache=cache
    )

    assert engine.call_count == 1
    assert first.cached is False
    assert second.cached is True
    assert second.claims == first.claims


def test_a_changed_body_misses_the_cache() -> None:
    engine = _fake(_payload(SUPERSESSION))
    cache = InMemoryExtractionCache()

    extract_file_claims(
        file=FILE, text=DOCUMENT, corpus_names=CORPUS, engine=engine, cache=cache
    )
    extract_file_claims(
        file=FILE, text=DOCUMENT + "An added line.\n", corpus_names=CORPUS,
        engine=engine, cache=cache,
    )

    assert engine.call_count == 2


def test_the_cache_key_is_bound_to_engine_identity() -> None:
    prompt = build_extraction_prompt(file=FILE, human_body=BODY, corpus_names=CORPUS)
    first = FakeExtractionEngine()
    second = FakeExtractionEngine()
    second.revision = "fake-v2"

    assert extraction_cache_key(engine=first, prompt=prompt) != extraction_cache_key(
        engine=second, prompt=prompt
    )


def test_the_cache_key_is_bound_to_the_prompt_revision() -> None:
    engine = FakeExtractionEngine()
    prompt = build_extraction_prompt(file=FILE, human_body=BODY, corpus_names=CORPUS)
    reworded = replace(prompt, revision="truth-extraction-prompt-v999")

    assert extraction_cache_key(engine=engine, prompt=prompt) != extraction_cache_key(
        engine=engine, prompt=reworded
    )


def test_corpus_extraction_resolves_targets_against_the_corpus_it_was_given() -> None:
    engine = FakeExtractionEngine({FILE: _payload(SUPERSESSION)})

    results = {result.file: result for result in extract_corpus_claims(DOCUMENTS, engine=engine)}

    assert set(results) == set(DOCUMENTS)
    assert results[FILE].claims[0].superseded == "archive_policy_2026-01-05.md"
    assert results["archive_policy_2026-01-05.md"].claims == ()


# --- the provider adapter: schema, identity, direction -----------------------------------------

#: Asymmetric on purpose. The superseding memo sorts BEFORE the one it supersedes, so a
#: direction bug that orders a pair by name instead of by role cannot pass by accident.
OLD_FILE = "zeta_retention_policy.md"
NEW_FILE = "alpha_retention_policy.md"
PAIR_DOCUMENTS = {
    OLD_FILE: "The original retention rule. Status: active.\n",
    NEW_FILE: f"This memo supersedes {OLD_FILE} after review. Status: deprecated.\n",
}


def _pair_graph() -> ReasoningGraphProjection:
    return build_reasoning_graph(
        [
            Chunk(file.split("_")[0], f"/corpus/{file}", text, {"file": file, "ord": 0})
            for file, text in PAIR_DOCUMENTS.items()
        ],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        include_text=True,
    )


def _pair_provider() -> ExtractedClaimProposalProvider:
    extractions = extract_corpus_claims(PAIR_DOCUMENTS, engine=DeterministicExtractionEngine())
    return ExtractedClaimProposalProvider(extractions)


def _extracted(graph: ReasoningGraphProjection, provider: ExtractedClaimProposalProvider):
    report = proposal_report(graph, model_provider=provider)
    assert report.provider_failures == (), report.provider_failures
    return tuple(
        proposal
        for proposal in (*report.proposals, *report.rejected_proposals)
        if proposal.provider_id == provider.provider_id
    )


def test_schema_version_is_two_now_that_relations_carry_validity_and_status() -> None:
    assert PROPOSAL_SCHEMA_VERSION == 2


def test_validity_and_status_are_their_own_relations_not_disguised_references() -> None:
    graph = _pair_graph()
    documents = {
        **PAIR_DOCUMENTS,
        NEW_FILE: PAIR_DOCUMENTS[NEW_FILE] + "Effective valid_from 2026-02-01 the rule changes.\n",
    }
    extractions = extract_corpus_claims(documents, engine=DeterministicExtractionEngine())
    relations = {
        proposal.proposed_relation
        for proposal in _extracted(graph, ExtractedClaimProposalProvider(extractions))
    }

    assert "declares_validity" in relations
    assert "declares_status" in relations
    assert "references" not in relations


def test_extracted_supersession_runs_from_the_superseded_to_the_superseding_document() -> None:
    graph = _pair_graph()

    supersessions = [
        proposal
        for proposal in _extracted(graph, _pair_provider())
        if proposal.proposed_relation == "supersedes"
    ]

    assert len(supersessions) == 1
    assert supersessions[0].subject_id == OLD_FILE
    assert supersessions[0].object_id == NEW_FILE


def test_extracted_supersession_matches_the_direction_of_the_deterministic_rule() -> None:
    graph = _pair_graph()
    rule_pairs = {
        (proposal.subject_id, proposal.object_id)
        for proposal in deterministic_inference_proposals(graph)
        if proposal.rule_id == "deterministic.direct_textual_reference"
    }
    extracted_pairs = {
        (proposal.subject_id, proposal.object_id)
        for proposal in _extracted(graph, _pair_provider())
        if proposal.proposed_relation == "supersedes"
    }

    assert rule_pairs
    assert rule_pairs <= extracted_pairs


def test_every_extracted_proposal_requires_a_human_review() -> None:
    graph = _pair_graph()

    proposals = _extracted(graph, _pair_provider())

    assert proposals
    assert {proposal.status for proposal in proposals} == {"requires_review"}
    assert {proposal.confidence for proposal in proposals} == {None}


def test_extracted_proposals_carry_the_engine_and_prompt_identity() -> None:
    graph = _pair_graph()
    provider = _pair_provider()

    proposals = _extracted(graph, provider)

    assert provider.provider_id == DETERMINISTIC_EXTRACTION_ENGINE_ID
    assert PROMPT_REVISION in provider.provider_revision
    assert {proposal.provider_revision for proposal in proposals} == {provider.provider_revision}


def test_extracted_proposals_cite_the_quote_the_claim_was_read_from() -> None:
    graph = _pair_graph()

    supersession = next(
        proposal
        for proposal in _extracted(graph, _pair_provider())
        if proposal.proposed_relation == "supersedes"
    )

    # Equality, not membership: `"" in anything` is True, so a substring assertion here would
    # pass against a proposal carrying no quote at all.
    assert supersession.metadata["quote"] == PAIR_DOCUMENTS[NEW_FILE].rstrip("\n")


def test_the_provider_boundary_recomputes_extracted_proposal_ids() -> None:
    class ForgingProvider(ExtractedClaimProposalProvider):
        def propose(self, graph, context):
            # Each forged id is DISTINCT. A single repeated id would be caught by duplicate
            # id detection instead, and the test would pass without the canonical id check
            # ever running.
            return tuple(
                replace(proposal, id=f"ip_forged{index:018d}")
                for index, proposal in enumerate(super().propose(graph, context))
            )

    extractions = extract_corpus_claims(PAIR_DOCUMENTS, engine=DeterministicExtractionEngine())
    report = proposal_report(_pair_graph(), model_provider=ForgingProvider(extractions))

    assert report.failure_matrix["malformed_output"] == 1
    assert all(
        proposal.provider_id != DETERMINISTIC_EXTRACTION_ENGINE_ID
        for proposal in report.proposals
    )


def test_repeated_identical_claims_collapse_to_one_proposal() -> None:
    doubled = dict(PAIR_DOCUMENTS)
    doubled[NEW_FILE] = PAIR_DOCUMENTS[NEW_FILE] * 2
    extractions = extract_corpus_claims(doubled, engine=DeterministicExtractionEngine())

    proposals = _extracted(_pair_graph(), ExtractedClaimProposalProvider(extractions))

    assert len({proposal.id for proposal in proposals}) == len(proposals)


def test_extractions_from_two_different_engines_cannot_share_one_provider_identity() -> None:
    mixed = (
        *extract_corpus_claims(PAIR_DOCUMENTS, engine=DeterministicExtractionEngine()),
        *extract_corpus_claims(PAIR_DOCUMENTS, engine=FakeExtractionEngine()),
    )

    with pytest.raises(ValueError, match="one engine identity"):
        ExtractedClaimProposalProvider(mixed)


# --- extraction is an ingest concern ------------------------------------------------------------


def test_the_query_planner_never_reaches_the_extraction_package() -> None:
    """`max_model_calls` defaults to 0 on the planner; this pins the same rule structurally.

    Run in a subprocess so an import another test already performed cannot mask it.
    """
    probe = (
        "import sys, recall.reasoning_planner;"
        " print('recall.truth_extraction' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "False", completed.stdout


def test_an_undeclared_extra_field_rejects_the_whole_batch() -> None:
    """The model supplies semantics only. A field nobody declared is a field nobody validates."""
    with pytest.raises(ExtractionBatchRejected, match="claim_shape"):
        _normalize(_payload({**SUPERSESSION, "confidence": "0.9"}))


# --- refusals the ladder must not turn into crashes ---------------------------------------------
#
# Every rung below refuses a REFUSAL that was escaping as an exception instead. The module
# docstring for `extract.py` claims "a refusal is a RESULT, never an exception that escapes";
# these pin that claim against the inputs that broke it.


def test_a_non_string_kind_rejects_the_batch_rather_than_raising_type_error() -> None:
    """`kind` reaches a dict membership test straight from model JSON. An unhashable value
    raised TypeError out of the ladder instead of refusing the batch."""
    with pytest.raises(ExtractionBatchRejected, match="claim_shape"):
        _normalize(_payload({"kind": [], "quote": SUPERSESSION["quote"]}))


def test_one_crashing_file_does_not_abort_the_rest_of_the_corpus() -> None:
    engine = FakeExtractionEngine({FILE: json.dumps({"claims": [{"kind": {}, "quote": "x"}]})})

    results = {result.file: result for result in extract_corpus_claims(DOCUMENTS, engine=engine)}

    assert set(results) == set(DOCUMENTS)
    assert results[FILE].batch_rejection is not None
    assert results[FILE].batch_rejection.rung == "claim_shape"


def test_json_nested_past_the_recursion_limit_rejects_the_batch(monkeypatch) -> None:
    """`RecursionError` is not a `ValueError`, so it escaped rung 1 and unwound through the
    caller's corpus loop. A real payload deep enough to trigger it (~100k nested arrays) also
    destabilises the test harness when the guard is off, so raise it directly instead: the
    property under test is that this exception type becomes a refusal, not that CPython's
    parser has a particular depth limit."""
    def deep(_raw: str) -> object:
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON object")

    monkeypatch.setattr("recall.truth_extraction._normalize.json.loads", deep)

    with pytest.raises(ExtractionBatchRejected, match="json"):
        _normalize(_payload(SUPERSESSION))


def test_an_unclosed_frontmatter_block_refuses_the_file_before_the_engine_runs() -> None:
    """`parse_frontmatter` hands an unclosed block back as body, which would make the metadata
    lines quotable prose and let a model justify `supersedes: X` with the string `supersedes: X`
    after all. Refuse the document instead, and do not pay for an engine call on it."""
    engine = _fake(_payload(SUPERSESSION))
    unclosed = "---\nsupersedes: archive_policy_2026-01-05.md\nvalid_from: 2026-02-01\n\n" + BODY

    result = extract_file_claims(file=FILE, text=unclosed, corpus_names=CORPUS, engine=engine)

    assert result.claims == ()
    assert result.batch_rejection is not None
    assert result.batch_rejection.rung == "unclosed_frontmatter"
    assert engine.call_count == 0


def test_a_closed_frontmatter_block_is_not_refused() -> None:
    engine = _fake(_payload(SUPERSESSION))

    result = extract_file_claims(file=FILE, text=DOCUMENT, corpus_names=CORPUS, engine=engine)

    assert result.batch_rejection is None
    assert engine.call_count == 1


def test_a_corpus_name_listed_twice_is_not_a_false_ambiguity() -> None:
    accepted, rejected = _normalize(_payload(SUPERSESSION), corpus_names=(*CORPUS, CORPUS[0]))

    assert rejected == ()
    assert accepted[0].superseded == "archive_policy_2026-01-05.md"


def test_a_supersession_target_outside_this_graph_generation_is_skipped() -> None:
    """Targets resolve against `corpus_names`, which can name files this graph has no node for.
    Emitting the proposal anyway gives `proposal_to_graph_edge` an empty `from_node_id`."""
    partial = build_reasoning_graph(
        [
            Chunk(
                "alpha", f"/corpus/{NEW_FILE}", PAIR_DOCUMENTS[NEW_FILE],
                {"file": NEW_FILE, "ord": 0},
            )
        ],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        include_text=True,
    )

    proposals = _extracted(partial, _pair_provider())

    assert [p for p in proposals if p.proposed_relation == "supersedes"] == []
    assert [p for p in proposals if p.proposed_relation == "declares_status"]


def test_a_markdown_thematic_break_is_not_mistaken_for_an_unclosed_block() -> None:
    """A memo opening with `---` as a horizontal rule has no metadata to protect. Refusing it
    loses genuine prose claims and records a valid document as malformed, so the rung must be
    scoped to blocks that actually carry a key `recall.frontmatter` recognises."""
    engine = DeterministicExtractionEngine()
    thematic = "---\n\n# Release notes\n\nThis release supersedes archive_policy_2026-01-05.md.\n"

    result = extract_file_claims(file=FILE, text=thematic, corpus_names=CORPUS, engine=engine)

    assert result.batch_rejection is None
    assert [claim.kind for claim in result.claims] == ["supersession"]


def test_a_thematic_break_document_that_merely_mentions_a_key_is_not_refused() -> None:
    """The block is the CONTIGUOUS run of key lines after the opening fence, not every line
    in the file. A memo that documents frontmatter, or quotes a key in an example, is not a
    memo with frontmatter, and refusing it loses the claims its prose really does state."""
    engine = DeterministicExtractionEngine()
    mentions_a_key = (
        "---\n"
        "\n"
        "# How to write frontmatter\n"
        "\n"
        "Put a line like\n"
        "\n"
        "    supersedes: archive_policy_2026-01-05.md\n"
        "\n"
        "at the top. This release supersedes archive_policy_2026-01-05.md.\n"
    )

    result = extract_file_claims(
        file=FILE, text=mentions_a_key, corpus_names=CORPUS, engine=engine
    )

    assert result.batch_rejection is None
    assert [claim.kind for claim in result.claims] == ["supersession"]


def test_a_blank_line_inside_an_unclosed_block_does_not_lose_the_refusal() -> None:
    """The run of a real block is not always uninterrupted: an unrecognised key, a blank line,
    or a key name with a space can sit above the one that matters. Stopping at the first of
    those hands `supersedes: X` back to the ladder as quotable prose, which is a false negative
    on the strongest guard in the design."""
    engine = _fake(_payload(SUPERSESSION))
    interrupted = (
        "---\n"
        "title: Retention policy\n"
        "\n"
        "supersedes: archive_policy_2026-01-05.md\n"
        "valid_from: 2026-02-01\n"
        "\n"
        "We keep records for seven years.\n"
    )

    result = extract_file_claims(file=FILE, text=interrupted, corpus_names=CORPUS, engine=engine)

    assert result.claims == ()
    assert result.batch_rejection is not None
    assert result.batch_rejection.rung == "unclosed_frontmatter"
    assert engine.call_count == 0


def test_a_key_name_the_pattern_does_not_recognise_does_not_lose_the_refusal() -> None:
    """`parse_frontmatter` has no notion of a well formed key: every line up to the closing
    fence is in the block. So the scan must not stop on a key shape it happens not to match,
    such as one with a space in its name, when the validity key sits below it."""
    engine = _fake(_payload(SUPERSESSION))
    odd_key_above = (
        "---\n"
        "Last updated: 2026-01-01\n"
        "supersedes: archive_policy_2026-01-05.md\n"
    )

    result = extract_file_claims(file=FILE, text=odd_key_above, corpus_names=CORPUS, engine=engine)

    assert result.batch_rejection is not None
    assert result.batch_rejection.rung == "unclosed_frontmatter"
    assert engine.call_count == 0


def test_a_markdown_heading_ends_the_block_even_when_it_contains_a_colon() -> None:
    """A heading is prose, and prose ends the block. Without that, a heading such as
    `# Retention: what changed` reads as a key line, the run continues past it, and a later
    prose line shaped like a key drags the whole memo into a refusal."""
    engine = _fake(_payload(SUPERSESSION))
    heading_with_colon = (
        "---\n"
        "# Retention: what changed\n"
        "valid_from: we never agreed one, see the thread\n"
        "\n"
        "This memo supersedes archive_policy_2026-01-05.md after the January review.\n"
    )

    result = extract_file_claims(
        file=FILE, text=heading_with_colon, corpus_names=CORPUS, engine=engine
    )

    assert result.batch_rejection is None
    assert engine.call_count == 1


def test_a_yaml_continuation_line_does_not_end_the_block() -> None:
    """A list item or an indented continuation carries no colon, so it is not a key line, but
    it is plainly still inside the block. Ending the scan there loses the refusal for every
    key below it, which is how `supersedes: X` becomes quotable prose again."""
    engine = _fake(_payload(SUPERSESSION))
    with_a_list = (
        "---\n"
        "title: Retention policy\n"
        "tags:\n"
        "  - policy\n"
        "  - retention\n"
        "supersedes: archive_policy_2026-01-05.md\n"
        "\n"
        "We keep records for seven years.\n"
    )

    result = extract_file_claims(file=FILE, text=with_a_list, corpus_names=CORPUS, engine=engine)

    assert result.batch_rejection is not None
    assert result.batch_rejection.rung == "unclosed_frontmatter"
    assert engine.call_count == 0
