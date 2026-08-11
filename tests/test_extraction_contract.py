"""`recall.extraction`: the model-backed prose extractor, and what it is not allowed to do.

The rule-based attempt is the measured prior and it is a failure, stated in `fix.py:10`:
"Measured on that corpus, it proposes ZERO edges. Four survived the mechanical rules and all four
were wrong on review." Its four errors were reported speech, superseding a *claim inside* the
target twice, and hedging. A model is being brought in because those four distinctions are
semantic, not syntactic — so the tests that matter are the ones about what reaches the model and
what the library refuses to believe when it answers.

Properties, one test each:

1.  **Leakage.** The prompt carries `human_body` and the corpus name list, and nothing else.
    Pinned against the bytes captured at the provider boundary rather than against the template,
    because a template that looks clean and a call that sends frontmatter anyway is exactly the
    defect worth catching, and only one of those two things is what the provider receives.
2.  **Frontmatter is outside the cache key.** Suppressing already-declared values is phase 2's
    job, through chunk metadata. Keeping frontmatter out of the key means an edit to it — very
    much including one this system just made — never re-invokes the model.
3.  **Determinism.** Temperature 0, pinned revision, and a cache keyed on
    `(sha256(human_body), model_id, model_revision, prompt_version)`.
4.  **The cache never lies.** `ClaimCache.put` refuses to overwrite a key with different content
    and raises, so an entry cannot be silently replaced under a stable proposal id.
5.  **`--recheck`** deliberately re-calls the model on cached keys and reports the mismatch rate.
    A non-zero rate means temperature 0 is not determinism for this provider and the cache is the
    only thing making runs reproducible — worth knowing before an eviction renumbers every id.
6.  **Every malformed output path is refused**, one `pytest.raises(match=<field>)` each, plus one
    test that a well-formed response is *not* refused.
7.  **Failures stay in band.** They map onto the existing `ProviderFailureKind` values; nothing
    escapes as an exception to the caller.
8.  **The prompt is frozen and hashed**, and the hash is recorded in every artifact.
"""
from __future__ import annotations

import json

import pytest

from recall.extraction import (
    PROMPT_SHA256,
    PROMPT_TEMPLATE,
    PROMPT_VERSION,
    ClaimCache,
    ProseClaimExtractor,
    build_prompt,
    claim_cache_key,
    resolve_claim_extractor,
)

CORPUS = frozenset({"old_thing_2026", "other_memo_2026", "new"})
BODY = "This decision supersedes old_thing_2026 after the review."


class _Client:
    """A stand-in for the OpenAI-compatible client that records what actually left the process.

    Recording the serialized request, not the arguments, is deliberate: the leakage assertion has
    to run against the same bytes the provider would see, or it is testing the caller's intent
    rather than the call.
    """

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.sent: list[bytes] = []
        self.kwargs: list[dict] = []

    def complete(self, *, model: str, messages: list[dict], temperature: float) -> dict:
        self.sent.append(json.dumps(messages, sort_keys=True).encode("utf-8"))
        self.kwargs.append({"model": model, "temperature": temperature})
        payload = self._responses.pop(0) if self._responses else '{"claims": []}'
        return {
            "text": payload,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "monetary_cost_usd": 0.00021,
        }


def _one_claim(obj: str = "old_thing_2026", evidence: str | None = None, **over) -> str:
    claim = {
        "relation": "supersedes",
        "object": obj,
        "evidence": evidence if evidence is not None else BODY,
        "confidence": 0.9,
    }
    claim.update(over)
    return json.dumps({"claims": [claim]})


def _extractor(client: _Client, cache: ClaimCache | None = None) -> ProseClaimExtractor:
    return ProseClaimExtractor(client=client, cache=cache)


# --- 1. leakage --------------------------------------------------------------------------------


def test_the_prompt_carries_only_the_human_body_and_the_corpus_names():
    client = _Client(_one_claim())
    _extractor(client).extract(BODY, CORPUS, subject="new")

    sent = client.sent[0].decode("utf-8")
    assert BODY in sent
    # Asserted on a name the BODY never mentions. A mutation that sent an empty corpus list
    # survived the earlier version of this test, because "old_thing_2026" also appears in the
    # prose — so the assertion was satisfied by the body and proved nothing about the list.
    assert "other_memo_2026" not in BODY, "the fixture must keep these two sources separable"
    assert "other_memo_2026" in sent, "the corpus name list never reached the provider"


def test_frontmatter_never_reaches_the_provider():
    """Pinned at the boundary. `human_body` strips the block; this asserts nothing put it back."""
    client = _Client(_one_claim())
    _extractor(client).extract(BODY, CORPUS, subject="new")

    sent = client.sent[0].decode("utf-8")
    for leaked in ("valid_from", "valid_until", "supersedes:", "---"):
        assert leaked not in sent, f"{leaked!r} reached the provider"


def test_no_reference_date_is_sent():
    """A reference date makes output nondeterministic and invites relative-date resolution —
    "superseded last March" becoming a concrete edge the author never wrote."""
    client = _Client(_one_claim())
    _extractor(client).extract(BODY, CORPUS, subject="new")

    sent = client.sent[0].decode("utf-8")
    assert "20" not in sent.replace("old_thing_2026", "").replace("other_memo_2026", ""), (
        "a year-like token reached the prompt outside the corpus names"
    )


def test_the_scope_rule_is_stated_in_the_prompt():
    """The one refusal the model is predicted to need. `fix.py:174`: "supersedes the <noun> in X"
    replaces part of X, while `supersedes:` demotes the whole predecessor."""
    assert "entire" in PROMPT_TEMPLATE or "whole" in PROMPT_TEMPLATE
    assert "part" in PROMPT_TEMPLATE


# --- 2 & 3. cache key --------------------------------------------------------------------------


def test_the_cache_key_covers_body_model_revision_and_prompt_version():
    base = claim_cache_key(BODY, model_id="m", model_revision="r", prompt_version="p")
    assert base == claim_cache_key(BODY, model_id="m", model_revision="r", prompt_version="p")
    assert base != claim_cache_key("other", model_id="m", model_revision="r", prompt_version="p")
    assert base != claim_cache_key(BODY, model_id="M", model_revision="r", prompt_version="p")
    assert base != claim_cache_key(BODY, model_id="m", model_revision="R", prompt_version="p")
    assert base != claim_cache_key(BODY, model_id="m", model_revision="r", prompt_version="P")


def test_a_cache_hit_does_not_call_the_model_again(tmp_path):
    cache = ClaimCache(tmp_path / "claims.sqlite3")
    client = _Client(_one_claim(), _one_claim())
    ex = _extractor(client, cache)

    first = ex.extract(BODY, CORPUS, subject="new")
    second = ex.extract(BODY, CORPUS, subject="new")

    assert len(client.sent) == 1, "the second call went to the provider"
    assert first == second


def test_temperature_is_zero():
    client = _Client(_one_claim())
    _extractor(client).extract(BODY, CORPUS, subject="new")
    assert client.kwargs[0]["temperature"] == 0


# --- 4. the cache refuses to be rewritten ------------------------------------------------------


def test_put_refuses_to_overwrite_a_key_with_different_content(tmp_path):
    """Proposal ids are content hashes over the claims. Letting an entry change under a stable
    key would renumber ids that other artifacts already reference."""
    cache = ClaimCache(tmp_path / "claims.sqlite3")
    cache.put("k", '{"claims": []}')
    with pytest.raises(ValueError, match="different content"):
        cache.put("k", '{"claims": [{"relation": "supersedes"}]}')


def test_put_is_idempotent_for_identical_content(tmp_path):
    """Non-over-rejection: refusing a *changed* value must not mean refusing a repeat."""
    cache = ClaimCache(tmp_path / "claims.sqlite3")
    cache.put("k", '{"claims": []}')
    cache.put("k", '{"claims": []}')
    assert cache.get("k") == '{"claims": []}'


# --- 5. --recheck ------------------------------------------------------------------------------


def test_recheck_recalls_the_model_and_reports_a_zero_mismatch_rate(tmp_path):
    cache = ClaimCache(tmp_path / "claims.sqlite3")
    client = _Client(_one_claim(), _one_claim())
    ex = _extractor(client, cache)
    ex.extract(BODY, CORPUS, subject="new")

    report = ex.recheck(BODY, CORPUS, subject="new")

    assert len(client.sent) == 2, "--recheck must actually re-call the provider"
    assert report.rechecked == 1
    assert report.mismatches == 0
    assert report.mismatch_rate == 0.0


def test_recheck_reports_a_mismatch_without_corrupting_the_cache(tmp_path):
    """A non-zero rate is the finding, not an error: it means temperature 0 is not determinism
    for this provider. The cached entry must survive, since it is what every existing proposal id
    was computed from."""
    cache = ClaimCache(tmp_path / "claims.sqlite3")
    client = _Client(_one_claim(), _one_claim(obj="other_memo_2026"))
    ex = _extractor(client, cache)
    first = ex.extract(BODY, CORPUS, subject="new")

    report = ex.recheck(BODY, CORPUS, subject="new")

    assert report.mismatches == 1
    assert report.mismatch_rate == 1.0
    assert ex.extract(BODY, CORPUS, subject="new") == first, "the cache was overwritten"


# --- 6. malformed output -----------------------------------------------------------------------


def test_output_that_is_not_json_is_refused():
    with pytest.raises(ValueError, match="json"):
        _extractor(_Client("I think it supersedes the old one!")).extract(
            BODY, CORPUS, subject="new")


def test_output_without_a_claims_key_is_refused():
    with pytest.raises(ValueError, match="claims"):
        _extractor(_Client('{"result": []}')).extract(BODY, CORPUS, subject="new")


def test_a_claim_without_a_relation_is_refused():
    payload = json.dumps({"claims": [{"object": "old_thing_2026", "evidence": BODY,
                                      "confidence": 0.9}]})
    with pytest.raises(ValueError, match="relation"):
        _extractor(_Client(payload)).extract(BODY, CORPUS, subject="new")


def test_an_unknown_relation_is_refused():
    with pytest.raises(ValueError, match="relation"):
        _extractor(_Client(_one_claim(relation="obsoletes"))).extract(BODY, CORPUS, subject="new")


def test_a_claim_naming_a_document_outside_the_corpus_is_refused():
    """The corpus name list is in the prompt so the model *can* only name real files; this is the
    check that it *did*. A hallucinated filename would resolve to nothing downstream, which is
    the SKIP-list noise `fix.py:62` says makes a tool worthless."""
    with pytest.raises(ValueError, match="object"):
        _extractor(_Client(_one_claim(obj="a_memo_nobody_wrote"))).extract(
            BODY, CORPUS, subject="new")


def test_a_claim_whose_evidence_is_not_verbatim_in_the_body_is_refused():
    """The evidence span is the reviewer's whole basis for judging. A paraphrase would put the
    model's words where the author's are supposed to be."""
    with pytest.raises(ValueError, match="evidence"):
        _extractor(_Client(_one_claim(evidence="it replaces the old one"))).extract(
            BODY, CORPUS, subject="new")


def test_a_claim_with_an_out_of_range_confidence_is_refused():
    with pytest.raises(ValueError, match="confidence"):
        _extractor(_Client(_one_claim(confidence=1.4))).extract(BODY, CORPUS, subject="new")


def test_a_claim_about_the_subject_itself_is_refused():
    with pytest.raises(ValueError, match="object"):
        _extractor(_Client(_one_claim(obj="new"))).extract(BODY, CORPUS, subject="new")


def test_a_well_formed_response_is_accepted():
    """Non-over-rejection. Eight refusals above are worth nothing if the valid case is refused."""
    claims = _extractor(_Client(_one_claim())).extract(BODY, CORPUS, subject="new")
    assert len(claims) == 1
    assert claims[0].relation == "supersedes"
    assert claims[0].object == "old_thing_2026"
    assert claims[0].evidence == BODY


def test_too_many_claims_raises_the_message_the_framework_maps_to_wrong_cardinality():
    """`_providers.py:227` keys `wrong_cardinality` off the substring "maximum is". Reusing the
    existing wording reaches that branch without adding a second failure-handling path."""
    payload = json.dumps({"claims": [json.loads(_one_claim())["claims"][0]] * 50})
    with pytest.raises(ValueError, match="maximum is"):
        _extractor(_Client(payload)).extract(BODY, CORPUS, subject="new")


# --- 7. provider metadata ----------------------------------------------------------------------


def test_provider_metadata_has_a_non_null_revision_and_cost():
    client = _Client(_one_claim())
    ex = _extractor(client)
    ex.extract(BODY, CORPUS, subject="new")

    meta = ex.last_provider_metadata
    assert meta is not None
    assert meta.model_revision is not None, "an unpinned revision makes the run unreproducible"
    assert meta.monetary_cost_usd is not None
    assert meta.to_dict()["model_revision"] is not None


# --- 8. the prompt is frozen and hashed --------------------------------------------------------


def test_the_prompt_hash_matches_the_template():
    import hashlib

    assert PROMPT_SHA256 == hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def test_the_prompt_hash_is_recorded_on_every_claim():
    """The precision prediction for this extractor is conditional on the prompt. An unhashed
    prompt makes that condition unverifiable after the fact, which turns the prediction into an
    unfalsifiable claim."""
    claims = _extractor(_Client(_one_claim())).extract(BODY, CORPUS, subject="new")
    assert claims[0].prompt_sha256 == PROMPT_SHA256
    assert claims[0].prompt_version == PROMPT_VERSION


def test_build_prompt_is_pure():
    a = build_prompt(BODY, CORPUS)
    b = build_prompt(BODY, CORPUS)
    assert a == b, "the prompt must not vary between two calls with identical input"


# --- the extra is absent by default ------------------------------------------------------------


def test_the_extractor_is_off_unless_the_env_var_is_set():
    """Mirrors `entailment.py:75`: unset means None, not a default-on network call."""
    assert resolve_claim_extractor({}) is None
    assert resolve_claim_extractor({"RECALL_EXTRACT": "0"}) is None


def test_a_non_boolean_opt_in_is_refused():
    with pytest.raises(ValueError, match="RECALL_EXTRACT"):
        resolve_claim_extractor({"RECALL_EXTRACT": "maybe"})


def test_the_module_imports_with_the_extra_absent(monkeypatch):
    """`openai` is installed here via the `bench` extra, so "it imports fine" proves nothing on
    this machine. Making the import genuinely fail is what tests the laziness."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def no_openai(name, *a, **k):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_openai)
    monkeypatch.delitem(__import__("sys").modules, "recall.extraction", raising=False)

    module = importlib.import_module("recall.extraction")
    assert module.resolve_claim_extractor({}) is None
    assert module.PROMPT_SHA256


def test_the_real_client_names_the_extra_when_the_sdk_is_missing(monkeypatch):
    import builtins

    from recall.extraction import _OpenAICompatChatClient

    real_import = builtins.__import__

    def no_openai(name, *a, **k):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(ImportError, match=r"recall\[extract\]"):
        _OpenAICompatChatClient()
