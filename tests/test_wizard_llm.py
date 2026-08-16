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
    client = _FakeClient(
        [
            _payload(
                ["a", "b", "c"],
                ["what is the read-through cache p99", "sourdough fermentation", "baroque counterpoint",
                 "tidal locking of moons"],
            )
        ]
    )
    entries = L.generate_llm(CORPUS, client=client, per_class=3)
    gap = [e["query"] for e in entries if not e["answerable"]]
    assert not any("read-through cache" in q for q in gap)
    assert len(gap) == 3


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
