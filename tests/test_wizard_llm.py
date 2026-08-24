"""The provider/key page, the live model catalogue, and the LLM query generator.

Three things are being pinned here, and only the third is about questions.

The **key page** must not exist when the user said data security is required. That answer is the
one the whole wizard hangs off: query generation sends sampled chunks of the corpus to whatever
endpoint is chosen, so offering a cloud provider after that answer would ship the user's data to
the thing the answer was meant to prevent.

The **catalogue** must be fetched, not hardcoded. `recall/setup.py:625` already predicted this in a
comment — "this is a static list inside a released artifact, and the provider's roster is not ours"
— and was right: its `gpt-4o-mini` default is two generations stale.

The **generator** must emit the same canonical shape as the offline one, because the caller does
not branch on which produced the set.
"""

from __future__ import annotations

import json
import sys

import pytest

from recall.wizard import llm as L
from recall.wizard.queryset import QuerySetError

CORPUS = [
    "# Retry policy\n\nRequests are retried on 5xx responses with exponential backoff.\n",
    "# Read-through cache\n\nThe cache reduced p99 latency from 840ms to 120ms.\n",
    "# Storage engine\n\nPostgres was chosen over a document store for a relational access "
    "pattern.\n",
]


class _FakeClient:
    """Records what it was asked and returns whatever it was told to."""

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def complete_json(self, *, system: str, user: str, schema: dict) -> dict:
        self.calls.append({"system": system, "user": user, "schema": schema})
        return self.payloads.pop(0)


def _payload(answerable: list[str], unanswerable: list[str]) -> dict:
    return {
        "answerable": [{"query": q} for q in answerable],
        "unanswerable": [{"query": q} for q in unanswerable],
    }


# --------------------------------------------------------------------------------------
# The security answer gates the whole page
# --------------------------------------------------------------------------------------


def test_no_cloud_provider_is_offered_when_security_is_required() -> None:
    """Query generation sends corpus chunks to the endpoint. That is the data the answer protects.

    Withheld entirely rather than shown as unavailable, matching `reasoning_provider_choices`:
    an uninstalled package is something the reader can go and fix, whereas this is a decision they
    already made, and re-offering it invites undoing it by accident.
    """
    choices = L.provider_choices(security_required=True, internet=True)
    values = {c.value for c in choices}
    assert values == {L.LOCAL_PROVIDER}


def test_cloud_providers_are_offered_when_security_is_not_required() -> None:
    values = {c.value for c in L.provider_choices(security_required=False, internet=True)}
    assert L.OPENROUTER_BASE_URL in values
    assert L.OPENAI_BASE_URL in values
    assert L.LOCAL_PROVIDER in values


def test_the_first_choice_is_always_runnable() -> None:
    """`recall.setup._choose` falls back to `choices[0]` and refuses a menu whose first entry is
    unavailable, so the local endpoint has to lead: it is the only one needing neither key nor
    network."""
    for security in (True, False):
        for internet in (True, False):
            choices = L.provider_choices(security_required=security, internet=internet)
            assert choices[0].available, (security, internet)
            assert choices[0].value == L.LOCAL_PROVIDER


def test_cloud_is_marked_unavailable_without_internet() -> None:
    choices = L.provider_choices(security_required=False, internet=False)
    cloud = [c for c in choices if c.value != L.LOCAL_PROVIDER]
    assert cloud and not any(c.available for c in cloud)
    assert all("internet" in c.unavailable_note for c in cloud)


# --------------------------------------------------------------------------------------
# The catalogue: live, with a pinned fallback
# --------------------------------------------------------------------------------------


def test_openrouter_catalogue_is_parsed_from_the_live_response(monkeypatch) -> None:
    body = json.dumps(
        {
            "data": [
                {
                    "id": "openai/gpt-5.6-luna",
                    "context_length": 1_050_000,
                    "pricing": {"prompt": "0.0000001", "completion": "0.0000006"},
                },
                {
                    "id": "some/embedding-model",
                    "context_length": 512,
                    "pricing": {"prompt": "0.00000002", "completion": "0"},
                },
                {
                    "id": "openrouter/auto",
                    "context_length": 2_000_000,
                    "pricing": {"prompt": "-1", "completion": "-1"},
                },
                {"id": "broken/entry-with-no-pricing"},
            ]
        }
    ).encode()
    monkeypatch.setattr(L, "_fetch", lambda url, **kw: body)

    models = L.openrouter_catalogue()
    ids = [m.id for m in models]
    assert "openai/gpt-5.6-luna" in ids
    # Zero completion price means it does not generate; it cannot author questions.
    assert "some/embedding-model" not in ids
    # NEGATIVE is the live roster's sentinel for a variable price, used by the router
    # pseudo-models. They pick a different underlying model per call, so a calibration measured
    # through one is bound to a model nobody can name. Found by fetching the real catalogue:
    # `if not completion` kept all of them, because -1.0 is truthy.
    assert "openrouter/auto" not in ids
    # One malformed entry must not lose the roster.
    assert "broken/entry-with-no-pricing" not in ids


def test_the_pinned_defaults_are_all_real_ids_in_the_live_roster() -> None:
    """A fallback list that has gone stale is worse than none: it names models that 404.

    Hits the live public catalogue, so it is skipped rather than failed when offline. This is the
    check that would have caught `gpt-4o-mini` going two generations stale in `recall/setup.py`.
    """
    models = L.openrouter_catalogue()
    if models == list(L.PINNED_OPENROUTER_MODELS):
        pytest.skip("no live catalogue available (offline or the API moved)")
    live = {m.id for m in models}
    missing = [m.id for m in L.PINNED_OPENROUTER_MODELS if m.id not in live]
    assert not missing, f"pinned models no longer on OpenRouter: {missing}"


def test_the_catalogue_falls_back_to_the_pinned_list_when_the_fetch_fails(monkeypatch) -> None:
    """A released artifact must still show a menu when the network is down or the API moved."""

    def boom(url, **kw):
        raise OSError("no route to host")

    monkeypatch.setattr(L, "_fetch", boom)
    models = L.openrouter_catalogue()
    assert [m.id for m in models] == [m.id for m in L.PINNED_OPENROUTER_MODELS]


def test_a_malformed_catalogue_falls_back_rather_than_raising(monkeypatch) -> None:
    monkeypatch.setattr(L, "_fetch", lambda url, **kw: b"<html>not json</html>")
    assert L.openrouter_catalogue() == list(L.PINNED_OPENROUTER_MODELS)


def test_the_default_model_leads_the_menu(monkeypatch) -> None:
    """`_choose` returns `choices[0]` on an unavailable pick, so the default has to be first."""
    monkeypatch.setattr(L, "_fetch", lambda url, **kw: b"nope")
    assert L.model_choices(L.OPENROUTER_BASE_URL)[0].value == L.DEFAULT_OPENROUTER_MODEL
    assert L.model_choices(L.OPENAI_BASE_URL)[0].value == L.DEFAULT_OPENAI_MODEL


def test_manual_entry_is_always_last(monkeypatch) -> None:
    """The mitigation for a roster that is not ours. It must survive on every provider."""
    monkeypatch.setattr(L, "_fetch", lambda url, **kw: b"nope")
    for base in (L.OPENROUTER_BASE_URL, L.OPENAI_BASE_URL):
        assert L.model_choices(base)[-1].value == L.MANUAL_MODEL


def test_no_prices_are_baked_into_the_menu_labels(monkeypatch) -> None:
    """`recall/setup.py:631` states the convention: a number in a shipped menu is a measurement
    nothing re-checks. Prices come from the live catalogue or not at all."""
    monkeypatch.setattr(L, "_fetch", lambda url, **kw: b"nope")
    for choice in L.model_choices(L.OPENROUTER_BASE_URL):
        assert "$" not in choice.description


# --------------------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------------------


def test_generated_entries_are_canonical_and_balanced() -> None:
    client = _FakeClient(
        [_payload([f"answerable {i}" for i in range(3)], [f"off topic {i}" for i in range(3)])]
    )
    entries = L.generate_llm(CORPUS, client=client, per_class=3)
    assert len(entries) == 6
    assert sum(1 for e in entries if e["answerable"]) == 3
    assert all(set(e) <= {"query", "answerable", "relevant_ids"} for e in entries)


def test_the_prompt_carries_real_chunks_not_a_summary() -> None:
    """A question written without seeing the corpus is not grounded in it."""
    client = _FakeClient([_payload(["a", "b", "c"], ["x", "y", "z"])])
    L.generate_llm(CORPUS, client=client, per_class=3)
    sent = client.calls[0]["user"]
    assert "read-through cache" in sent.lower()
    assert "840ms" in sent


def test_the_model_is_told_the_gap_class_must_be_a_different_domain() -> None:
    """The measured constraint. A near-miss gap class is not separable.

    `recall/eval/synthetic.py:69-75`: perturbing an answerable query gave median top cosine 0.830
    against answerable 0.923, with nothing below the answerable floor. The instruction has to say
    so, or the model will produce plausible near-misses, which is the natural thing to write.
    """
    client = _FakeClient([_payload(["a", "b", "c"], ["x", "y", "z"])])
    L.generate_llm(CORPUS, client=client, per_class=3)
    system = client.calls[0]["system"].lower()
    assert "different" in system or "unrelated" in system
    assert "not" in system


def test_a_short_response_is_refused_rather_than_padded() -> None:
    """Certification needs both classes at the floor; a thin set must fail here, not there."""
    client = _FakeClient([_payload(["only one"], ["a", "b", "c"])])
    with pytest.raises(QuerySetError):
        L.generate_llm(CORPUS, client=client, per_class=3)


def test_duplicates_from_the_model_are_removed() -> None:
    client = _FakeClient([_payload(["same", "same", "other", "third"], ["x", "y", "z"])])
    entries = L.generate_llm(CORPUS, client=client, per_class=3)
    answerable = [e["query"] for e in entries if e["answerable"]]
    assert len(set(answerable)) == 3


def test_a_gap_query_the_model_copied_from_the_corpus_is_dropped() -> None:
    """The model is asked for a disjoint domain and will sometimes ignore it.

    Enforced rather than trusted, for the same reason the offline generator checks its subject
    list against the corpus: a gap class sharing the corpus vocabulary is the failure that made an
    earlier abstention measurement meaningless.
    """
    # Drawn from the pool rather than spelled out: a literal subject in a test file joins the
    # corpus whenever a code corpus is rooted at this repository, and the filter then disqualifies
    # it for every user. That is precisely how the pool came to disqualify all 25 of its own
    # subjects when it lived as Python literals.
    from recall.eval.synthetic import _OFFTOPIC_SUBJECTS

    client = _FakeClient(
        [
            _payload(
                ["a", "b", "c"],
                ["what is the read-through cache p99", *_OFFTOPIC_SUBJECTS[1:4]],
            )
        ]
    )
    entries = L.generate_llm(CORPUS, client=client, per_class=3)
    gap = [e["query"] for e in entries if not e["answerable"]]
    assert not any("read-through cache" in q for q in gap)
    assert len(gap) == 3


def test_the_vocabulary_filter_agrees_with_the_offline_generators_own_disjointness_test() -> None:
    """Two independent judgements of "does this subject appear in the corpus" must not disagree.

    This is the test that would have caught the filter's first version. Against a REALISTIC corpus
    it rejected 83 of the 125 questions in the project's own certified-disjoint off-topic pool,
    because it intersected every corpus token filtered only by `len(w) > 4` — so "should",
    "before" and "engine" were all treated as subject matter, and `generate_llm` raised on any
    real corpus after the paid model call. A three-chunk fixture with 24 word types cannot show
    that; a corpus with real English can.

    The invariant: a pool question is rejected exactly when `offtopic_subjects_absent_from` says
    its subject is not disjoint from the corpus.
    """
    from pathlib import Path

    from recall.eval.synthetic import _OFFTOPIC_SUBJECTS, _OFFTOPIC_TEMPLATES

    from recall.wizard.queryset import chunks_from_directory, offtopic_subjects_absent_from

    # This repository's own `docs/`, deliberately. A synthetic fixture cannot exercise the defect:
    # the first version of the filter looked fine on three short chunks and failed on real prose,
    # and a fixture built by repeating a chunk makes every term look ubiquitous, which sends the
    # df ceiling degenerate rather than testing it. `docs/` is also the corpus the ceiling was
    # tuned against, and it genuinely discusses 12 of the 25 off-topic subjects, so both the
    # accept and the reject path are real.
    docs = Path(__file__).resolve().parents[1] / "docs"
    if not docs.is_dir():
        pytest.skip("docs/ is not present in this checkout")
    corpus = chunks_from_directory(docs)

    disjoint = set(offtopic_subjects_absent_from(corpus))
    subject_words = L._corpus_subject_words(corpus)

    disagreements = []
    for subject in _OFFTOPIC_SUBJECTS:
        for template in _OFFTOPIC_TEMPLATES:
            query = template.format(s=subject)
            rejected = L._reuses_corpus_vocabulary(query, subject_words)
            if rejected != (subject not in disjoint):
                disagreements.append((query, rejected, subject in disjoint))

    assert not disagreements, f"filters disagree on {len(disagreements)} question(s): {disagreements[:3]}"
    # Both paths really were exercised, so this cannot pass by rejecting or accepting everything.
    assert 0 < len(disjoint) < len(_OFFTOPIC_SUBJECTS)


def test_an_on_topic_question_is_still_caught() -> None:
    """Loosening the filter must not stop it doing its job. All four were measured as caught."""
    from pathlib import Path

    from recall.wizard.queryset import chunks_from_directory

    docs = Path(__file__).resolve().parents[1] / "docs"
    if not docs.is_dir():
        pytest.skip("docs/ is not present in this checkout")
    subject_words = L._corpus_subject_words(chunks_from_directory(docs))

    for question in (
        "how do i tune the answerability floor for my corpus",
        "what happens when a generation is promoted before it is calibrated",
        "which postgres extension does the index require",
        "how long does indexing take for a thousand documents",
    ):
        assert L._reuses_corpus_vocabulary(question, subject_words), question


def test_a_corpus_of_near_identical_chunks_does_not_disable_the_filter() -> None:
    """An empty subject set makes `_reuses_corpus_vocabulary` return False for everything.

    That is a guard that stops guarding while every test feeding it a large corpus still passes.
    At a 5% ceiling with no floor, a corpus whose chunks resemble each other produced exactly
    that.
    """
    corpus = ["# Note\n\nThe replication topology tolerates a single regional failure.\n"] * 30
    subject_words = L._corpus_subject_words(corpus)
    assert subject_words, "an empty subject set silently disables the filter"
    assert L._reuses_corpus_vocabulary(
        "how does the replication topology tolerate a regional failure", subject_words
    )


def test_the_prompt_is_bounded_by_characters_not_chunk_count() -> None:
    """A count-based bound says nothing about size: 120 chunks measured 75k characters on real
    docs, against the 8k the docstring claimed, and more than a local endpoint's context.

    `per_class` must be large enough that the budget actually binds. At 3 the sample is 9 chunks,
    about 7 200 characters, which is under the budget however the code behaves — so an earlier
    version of this test passed with the budget check deleted.
    """
    per_class = 20
    fat = ["x" * 800 for _ in range(500)]
    unbounded = per_class * L.CHUNKS_PER_PROMPT_MULTIPLIER * 800
    assert unbounded > L.MAX_PROMPT_CHARS, "precondition: the budget must actually bind here"

    client = _FakeClient(
        [_payload([f"a{i}" for i in range(40)], [f"zzz{i}" for i in range(40)])]
    )
    L.generate_llm(fat, client=client, per_class=per_class)
    sent = client.calls[0]["user"]
    assert len(sent) <= L.MAX_PROMPT_CHARS + 500, len(sent)  # + the instruction text


def test_the_excerpts_span_the_corpus_rather_than_its_head() -> None:
    """Trimming a SORTED sample from the tail keeps only its lowest-indexed part.

    The same mistake as the offline generator's sampling loop, made a second time in this package.
    Measured on this repository's `docs/` at the default: sorted-and-break showed the model 39
    chunks from 3 of 51 files, sampled-and-continue 40 chunks from 21 of 51, on the same budget.
    A question set written from three documents is not a question set about the corpus.

    Measured over DOCUMENTS, not over index range. Chunks from one document are contiguous, so the
    bias shows up as "three files" long before it shows up as "low indices" — an earlier version of
    this test asserted `max(index) > 200` and passed against the buggy code, because a 60-chunk
    sample from 400 still reaches index 200 once sorted.

    The chunk size is chosen so only about a third of the sample fits the budget, matching the
    ratio measured on the real corpus (39 of 120). That ratio is what makes the head bias bite.
    """
    docs, per_doc = 20, 20
    corpus = [f"doc{_token(d)} part{_token(p)} " + "y" * 1240 for d in range(docs) for p in range(per_doc)]
    assert L.MAX_PROMPT_CHARS // len(corpus[0]) < 20 * L.CHUNKS_PER_PROMPT_MULTIPLIER // 2

    client = _FakeClient(
        [_payload([f"a{i}" for i in range(40)], [f"zzz{i}" for i in range(40)])]
    )
    L.generate_llm(corpus, client=client, per_class=20, seed=0)

    sent = client.calls[0]["user"]
    reached = {d for d in range(docs) if f"doc{_token(d)} " in sent}
    # 10, measured rather than chosen. Swept over 200 seeds against this exact selection logic:
    # sorted-and-break reaches at most 9 documents (mean 6.5), sampled-and-continue at least 9
    # (mean 12.6). The two distributions do not overlap at 10. An earlier threshold of 12 sat on
    # the fixed code's own mean, so 18.5% of seeds would have turned a CORRECT implementation red
    # the moment the budget, the multiplier or the padding changed.
    assert len(reached) >= 10, f"excerpts came from only {len(reached)} of {docs} documents"


def _token(i: int) -> str:
    """A unique alphabetic marker per chunk index, with no digits to disturb tokenisation."""
    syllables = ("ka", "lo", "mi", "ru", "ta", "ne", "vo", "shi", "pe", "du")
    return syllables[i // 100 % 10] + syllables[i // 10 % 10] + syllables[i % 10]


def test_an_oversized_chunk_is_skipped_rather_than_ending_the_selection() -> None:
    """`break` let one long chunk truncate everything after it in the iteration order.

    Ten oversized chunks interleaved with ten small ones, so the result does not depend on where a
    single giant happens to land in the sampled permutation — an earlier version used one giant
    and passed or failed by luck of the seed.
    """
    corpus: list[str] = []
    for i in range(10):
        corpus.append(f"small{_token(i)} " + "a" * 100)
        corpus.append("x" * (L.MAX_PROMPT_CHARS + 10))

    client = _FakeClient([_payload([f"a{i}" for i in range(9)], [f"zzz{i}" for i in range(9)])])
    L.generate_llm(corpus, client=client, per_class=6, seed=0)

    sent = client.calls[0]["user"]
    kept = sum(1 for i in range(10) if f"small{_token(i)} " in sent)
    # Every small chunk sampled must survive: they all fit, and skipping a giant costs nothing.
    # Stopping at the first giant leaves at most a couple.
    assert kept >= 7, f"only {kept} of the small chunks reached the prompt"
    assert "x" * 200 not in sent, "an oversized chunk was included"


def test_the_model_menu_is_short_enough_to_read(monkeypatch) -> None:
    """`recall.setup._choose` prints every entry as a numbered line before prompting, and the live
    OpenRouter roster is ~390 models."""
    many = [
        L.CatalogueModel(f"vendor/model-{i}", completion_price=1e-7 * i) for i in range(400)
    ]
    monkeypatch.setattr(L, "openrouter_catalogue", lambda: many)
    choices = L.model_choices(L.OPENROUTER_BASE_URL)
    assert len(choices) == L.MENU_LIMIT + 1  # + manual entry
    assert choices[-1].value == L.MANUAL_MODEL


def test_a_non_list_data_field_falls_back_rather_than_raising(monkeypatch) -> None:
    """`{"data": null}` is an ordinary gateway error shape and used to escape as a TypeError."""
    for body in (b'{"data": null}', b'{"data": 5}', b'{"data": {"a": 1}}'):
        monkeypatch.setattr(L, "_fetch", lambda url, _b=body, **kw: _b)
        assert L.openrouter_catalogue() == list(L.PINNED_OPENROUTER_MODELS)


def test_a_pasted_key_with_a_trailing_newline_still_works(monkeypatch) -> None:
    """Copying a key out of a terminal brings a trailing newline, which is header-illegal."""
    seen: dict[str, str] = {}

    def capture(url, *, headers=None, **kw):
        seen.update(headers or {})
        return json.dumps({"data": [{"id": "gpt-5.6-luna"}]}).encode()

    monkeypatch.setattr(L, "_fetch", capture)
    L.openai_catalogue("sk-secret-ABC123\n")
    assert seen["Authorization"] == "Bearer sk-secret-ABC123"


def test_an_api_key_never_reaches_a_log_line(caplog) -> None:
    """`http.client` raises ValueError("Invalid header value %r" % value) whose value is the whole
    `Bearer <key>` header, and logging `%s` of that exception wrote the key into a warning.

    The key here carries an EMBEDDED newline, not a trailing one. A trailing newline is removed by
    the `.strip()` above, so the exception never fires and this test would pass with the log
    redaction deleted — which is exactly what it did until the guard was mutated. Embedded is the
    case `.strip()` cannot fix, so it is the case that exercises the redaction.
    """
    import logging

    secret = "sk-secret-ABC123"
    with caplog.at_level(logging.DEBUG):
        models = L.openai_catalogue(f"{secret}\nEXTRA")
    assert models == list(L.PINNED_OPENAI_MODELS), "the bad header must fall back, not raise"
    assert secret not in caplog.text, caplog.text


def test_the_openai_roster_excludes_models_that_cannot_chat(monkeypatch) -> None:
    """Offering `text-embedding-3-small` as the thing that writes your questions is the same
    defect the OpenRouter path was fixed for."""
    body = json.dumps(
        {
            "data": [
                {"id": "gpt-5.6-luna"},
                {"id": "text-embedding-3-small"},
                {"id": "whisper-1"},
                {"id": "dall-e-3"},
                {"no_id": True},
                {"id": "gpt-5.6-terra"},
            ]
        }
    ).encode()
    monkeypatch.setattr(L, "_fetch", lambda url, **kw: body)
    ids = [m.id for m in L.openai_catalogue("sk-test")]
    assert ids == ["gpt-5.6-luna", "gpt-5.6-terra"]


# --------------------------------------------------------------------------------------
# The concrete client
# --------------------------------------------------------------------------------------


class _FakeCompletions:
    """Stands in for `client.chat.completions`, recording the response_format it was given."""

    def __init__(self, script) -> None:
        self.script = script
        self.formats: list[str] = []

    def create(self, **kwargs):
        self.formats.append((kwargs.get("response_format") or {}).get("type", "none"))
        outcome = self.script(kwargs)
        if isinstance(outcome, Exception):
            raise outcome

        class _M:
            content = outcome

        class _C:
            message = _M()

        class _R:
            choices = [_C()]

        return _R()


def _client_with(script) -> tuple[L.OpenAICompatClient, _FakeCompletions]:
    client = L.OpenAICompatClient.__new__(L.OpenAICompatClient)
    client.model = "test-model"
    fake = _FakeCompletions(script)

    class _Chat:
        completions = fake

    class _Inner:
        chat = _Chat()

    client._client = _Inner()
    return client, fake


def test_the_client_asks_for_a_json_schema_first() -> None:
    """Structured output is requested, not asked for in prose."""
    client, fake = _client_with(lambda kw: '{"answerable": [], "unanswerable": []}')
    assert client.complete_json(system="s", user="u", schema=_SCHEMA_STUB) == {
        "answerable": [],
        "unanswerable": [],
    }
    assert fake.formats == ["json_schema"]


def test_an_endpoint_that_rejects_json_schema_falls_back() -> None:
    """A local server without schema support must still work rather than be unusable."""

    def script(kw):
        fmt = (kw.get("response_format") or {}).get("type")
        if fmt == "json_schema":
            return ValueError("response_format.json_schema is not supported")
        return '{"answerable": [], "unanswerable": []}'

    client, fake = _client_with(script)
    client.complete_json(system="s", user="u", schema=_SCHEMA_STUB)
    assert fake.formats == ["json_schema", "json_object"]


def test_a_fenced_response_is_still_parsed() -> None:
    """Models add ```json despite being told to return raw JSON."""
    client, _ = _client_with(lambda kw: '```json\n{"answerable": [], "unanswerable": []}\n```')
    assert client.complete_json(system="s", user="u", schema=_SCHEMA_STUB) == {
        "answerable": [],
        "unanswerable": [],
    }


def test_unparseable_output_from_every_format_raises_rather_than_returning_empty() -> None:
    """Returning {} would surface much later as a thin query set blamed on the corpus."""
    client, fake = _client_with(lambda kw: "I'm sorry, I can't help with that.")
    with pytest.raises(QuerySetError, match="nothing usable"):
        client.complete_json(system="s", user="u", schema=_SCHEMA_STUB)
    assert fake.formats == ["json_schema", "json_object", "none"]


_SCHEMA_STUB = {"type": "object", "properties": {}}


class _Unauthorized(Exception):
    """Named to match what the openai SDK raises, which is how `_is_unrecoverable` classifies."""


_Unauthorized.__name__ = "AuthenticationError"


def test_an_auth_failure_is_not_retried_under_every_response_format() -> None:
    """A 401 is not fixed by a different response_format.

    Retrying it three times costs three round trips against `max_retries=0`, and the final message
    would be about model output, pointing the user at their corpus rather than at their key.
    """
    client, fake = _client_with(lambda kw: _Unauthorized("Incorrect API key provided"))
    with pytest.raises(QuerySetError, match="endpoint refused"):
        client.complete_json(system="s", user="u", schema=_SCHEMA_STUB)
    assert fake.formats == ["json_schema"], "must fail on the first attempt, not the third"


def test_a_rate_limit_does_not_tell_the_user_to_check_their_key() -> None:
    """A 429 is the one transient member of the unrecoverable set.

    It must still fail fast rather than burn three round trips, but sending a throttled user to
    rotate a key that was never wrong is the wrong instruction.
    """

    class _Throttled(Exception):
        pass

    _Throttled.__name__ = "RateLimitError"

    client, fake = _client_with(lambda kw: _Throttled("429 Too Many Requests"))
    with pytest.raises(QuerySetError) as exc:
        client.complete_json(system="s", user="u", schema=_SCHEMA_STUB)
    assert "rate-limited" in str(exc.value)
    assert "Check the API key" not in str(exc.value)
    assert fake.formats == ["json_schema"], "still fails fast, not three times"


def test_the_api_key_cannot_reach_the_final_error_message() -> None:
    """`{last!r}` put the raw exception repr into the message, undoing `_safe_reason`.

    httpx raises with the whole `Bearer <key>` header in its message for a key carrying an
    interior invalid character, which `.strip()` cannot remove.
    """
    secret = "sk-secret-ABC123"
    client, _ = _client_with(lambda kw: ValueError(f"Invalid header value b'Bearer {secret}'"))
    with pytest.raises(QuerySetError) as exc:
        client.complete_json(system="s", user="u", schema=_SCHEMA_STUB)
    assert secret not in str(exc.value), str(exc.value)


@pytest.mark.parametrize("choices", [[], None])
def test_a_response_with_no_choices_stays_in_the_declared_exception_type(choices) -> None:
    """OpenRouter returns HTTP 200 with an error body and no choices for an upstream failure.

    What matters is the LINE PLACEMENT, not the explicit emptiness check: reading
    `completion.choices[0]` outside the parse `try` let an IndexError (empty list) or TypeError
    (None) escape as a raw exception and skip the fallback entirely. Asserting `QuerySetError`
    over all three formats is what distinguishes the two, because with the read inside the try the
    loop absorbs it and exhausts the formats normally.

    An earlier version of this test deleted only the `if not choices` line, which changes nothing:
    `choices[0]` raises anyway and the same `except` catches it. The test stayed green and proved
    nothing.
    """

    class _NoChoices:
        pass

    _NoChoices.choices = choices

    client, fake = _client_with(lambda kw: _NoChoices())
    with pytest.raises(QuerySetError, match="nothing usable"):
        client.complete_json(system="s", user="u", schema=_SCHEMA_STUB)
    assert fake.formats == ["json_schema", "json_object", "none"], "the fallback must still run"


@pytest.mark.parametrize("body", ["[]", '[["a","b"]]', "[1,2]", '"hello"', "null"])
def test_non_object_json_is_refused_rather_than_coerced(body) -> None:
    """`dict(json.loads("[]"))` is `{}` and succeeds — the empty dict this class must never return.

    `'[["a","b"]]'` is worse: it coerces into a fabricated `{"a": "b"}`.
    """
    client, _ = _client_with(lambda kw: body)
    with pytest.raises(QuerySetError, match="nothing usable"):
        client.complete_json(system="s", user="u", schema=_SCHEMA_STUB)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('```json {"a": 1}```', {"a": 1}),
        ('```{"a": 1}```', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('{"a": 1}', {"a": 1}),
    ],
)
def test_every_fence_shape_a_model_emits_is_unwrapped(raw, expected) -> None:
    """A single-line fence had no newline after the opener, so the payload was discarded whole."""
    client, _ = _client_with(lambda kw: raw)
    assert client.complete_json(system="s", user="u", schema=_SCHEMA_STUB) == expected


def test_the_constructor_configures_the_sdk_client(monkeypatch) -> None:
    """No test executed `__init__` at all, so a wrong kwarg would have shipped green."""
    seen: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    fake_module = type(sys)("openai")
    fake_module.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    L.OpenAICompatClient(
        base_url="http://localhost:11434/v1", api_key="  sk-padded  ", model="m", timeout=9.0
    )
    assert seen["api_key"] == "sk-padded", "a pasted key's whitespace must be stripped"
    assert seen["max_retries"] == 0, "a retry re-sends thousands of tokens"
    assert seen["base_url"] == "http://localhost:11434/v1"
    assert seen["timeout"] == 9.0


def test_a_missing_openai_extra_says_how_to_fix_it(monkeypatch) -> None:
    """A bare ModuleNotFoundError reads as a library bug rather than an uninstalled extra."""
    import builtins

    real_import = builtins.__import__

    def deny(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny)
    with pytest.raises(ImportError, match=r"recall-rag\[extract\]"):
        L.OpenAICompatClient(base_url="x", api_key="k", model="m")


def test_the_schema_forces_the_two_classes_apart() -> None:
    """Structured output, not free-text parsing: a malformed batch must fail loudly rather than
    silently yield fewer queries than the certification floor needs."""
    client = _FakeClient([_payload(["a", "b", "c"], ["x", "y", "z"])])
    L.generate_llm(CORPUS, client=client, per_class=3)
    schema = client.calls[0]["schema"]
    assert set(schema["properties"]) == {"answerable", "unanswerable"}
    assert schema["required"] == ["answerable", "unanswerable"]


def test_chunk_sampling_is_deterministic_for_a_seed() -> None:
    """The registered measurement re-runs this arm; the prompt must not drift between runs."""
    a = _FakeClient([_payload(["a", "b", "c"], ["x", "y", "z"])])
    b = _FakeClient([_payload(["a", "b", "c"], ["x", "y", "z"])])
    L.generate_llm(CORPUS, client=a, per_class=3, seed=5)
    L.generate_llm(CORPUS, client=b, per_class=3, seed=5)
    assert a.calls[0]["user"] == b.calls[0]["user"]
