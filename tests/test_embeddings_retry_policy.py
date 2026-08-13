"""One retry policy per request, and exactly one, on the corpus indexing path.

``retry_with_backoff`` owns retries for the cloud embedders. Every SDK ships its own retry layer
underneath, so unless that layer is switched off the two multiply: a policy that says "three
attempts" spends nine requests, and the outer full-jitter backoff (which exists so a fleet does
not remarch onto the provider in lockstep) ends up wrapping an inner loop that has no jitter at
all.

This is the more expensive place for that arithmetic to live. ``OpenAICompatEmbedder`` is not a
benchmark path: it is what indexes a user's corpus, batch after batch, which is the traffic shape
a provider rate-limits. Answering a 429 with triple the requests aims the multiplication at a
provider that has just said it is overloaded, and triples the bill for it.

These tests count real HTTP requests rather than asserting on constructor keywords, because the
keyword is the mechanism and the request count is the behaviour. A fake ``embeddings.create``
would be blind to the layer under test: the SDK's retries live below that method, inside its
transport.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import recall.embeddings
from recall.embeddings import OpenAICompatEmbedder, VoyageEmbedder
from tests.provider_stub_helpers import EMBEDDINGS_OK, ProviderStub, provider_stub

openai = pytest.importorskip("openai")

#: Imported at module scope, not inside the test that needs it. `voyageai` pulls `transformers`
#: behind it and takes ~75s to import cold, which is most of this suite's 120s per-test timeout;
#: and `timeout_method = "thread"` takes the whole session down rather than one test. Collection
#: is not timed, so paying for it here is both faster and safer. `importorskip` cannot be used:
#: voyage is an optional extra, and skipping the module for its absence would also skip the
#: OpenAI-compatible tests below, which are the ones covering the live defect.
try:
    import voyageai
except ImportError:  # pragma: no cover - exercised only without the extra
    voyageai = None  # type: ignore[assignment]


@pytest.fixture
def instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the outer full-jitter sleep to zero so the retry paths run at test speed.

    ``retry_with_backoff`` takes an injectable ``sleep``, but ``_embed_one_batch`` does not pass
    one, so the draw itself is the only seam. Patching ``random.uniform`` rather than the sleep
    also leaves the call to ``sleep`` in place, which keeps the test honest about the number of
    delays the policy takes.
    """
    monkeypatch.setattr(recall.embeddings.random, "uniform", lambda _a, _b: 0.0)


def _armed(stub: ProviderStub, status: int, message: str) -> OpenAICompatEmbedder:
    """Build an embedder against the stub, then arm the stub to fail and reset the counter."""
    emb = OpenAICompatEmbedder(api_key="test", base_url=stub.base_url, batch_size=1)
    stub.track(emb._client)
    assert stub.count == 1, "construction should cost exactly one dim probe"
    stub.arm(status, message)
    return emb


def test_a_rate_limit_costs_the_retry_policys_attempts_and_no_more(instant_backoff: None) -> None:
    """One 429 must cost 3 requests, which is what ``max_retries=3`` means.

    With the SDK's own 2 retries left on, the layers multiply to 9: triple the load onto a
    provider that has just said it is overloaded, and triple the bill.
    """
    with provider_stub(EMBEDDINGS_OK) as stub:
        emb = _armed(stub, 429, "please slow down")

        with pytest.raises(openai.RateLimitError):
            emb.embed(["x"])

        assert stub.count == 3


def test_an_upstream_server_error_costs_the_retry_policys_attempts_and_no_more(
    instant_backoff: None,
) -> None:
    """Same arithmetic for 5xx. 507 is used deliberately: it is not one of the literal markers in
    the text fallback, so only the numeric branch can classify it transient."""
    with provider_stub(EMBEDDINGS_OK) as stub:
        emb = _armed(stub, 507, "insufficient storage upstream")

        with pytest.raises(openai.InternalServerError):
            emb.embed(["x"])

        assert stub.count == 3


def test_the_configured_attempt_budget_is_what_is_spent(instant_backoff: None) -> None:
    """``max_retries`` must mean requests, not requests times whatever the SDK defaults to.

    Pinning a second budget catches a fix that happens to make one number come out right. Under
    the defect this arm costs 15 rather than 5.
    """
    with provider_stub(EMBEDDINGS_OK) as stub:
        emb = OpenAICompatEmbedder(
            api_key="test", base_url=stub.base_url, batch_size=1, max_retries=5
        )
        stub.track(emb._client)
        stub.arm(429, "please slow down")

        with pytest.raises(openai.RateLimitError):
            emb.embed(["x"])

        assert stub.count == 5


def test_a_bad_request_is_not_retried(instant_backoff: None) -> None:
    """A permanent 4xx must still fail fast, at one request.

    The SDK does not retry 400 either, so this arm passes with the defect present. It is here to
    prove the fix removed a multiplication rather than the retry itself.
    """
    with provider_stub(EMBEDDINGS_OK) as stub:
        emb = _armed(stub, 400, "input is not valid for this model")

        with pytest.raises(openai.BadRequestError):
            emb.embed(["x"])

        assert stub.count == 1


def test_a_success_costs_exactly_one_request_per_batch(instant_backoff: None) -> None:
    """The happy path must not be perturbed: two texts at ``batch_size=1`` are two requests."""
    with provider_stub(EMBEDDINGS_OK) as stub:
        emb = OpenAICompatEmbedder(api_key="test", base_url=stub.base_url, batch_size=1)
        stub.track(emb._client)
        stub.count = 0

        vectors = emb.embed(["x", "y"])

        assert stub.count == 2
        assert vectors == [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]


def test_the_voyage_client_is_built_with_the_sdk_retry_layer_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Voyage's SDK defaults ``max_retries`` to 0, so this pins a default rather than fixing a
    live bug. Pinning it means a future SDK release that starts retrying cannot silently
    reintroduce the multiplication ``OpenAICompatEmbedder`` had.

    A fake client is honest here in a way it would not be above: the assertion is about the
    keyword this repository passes, not about what the transport underneath does with it.
    """
    if voyageai is None:
        pytest.skip("needs the voyage extra (pip install recall[voyage])")
    seen: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

        def embed(self, texts: list[str], model: str | None = None) -> object:
            return SimpleNamespace(embeddings=[[0.0, 1.0]])

    monkeypatch.setattr(voyageai, "Client", _FakeClient)

    VoyageEmbedder(api_key="k")

    assert seen.get("max_retries") == 0, f"voyage client built with {seen!r}"
