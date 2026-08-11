from __future__ import annotations

import pytest

from benchmarks.artifact_contract import reject_unauditable_cost_claims
from recall.provider_metadata import ProviderMetadata


def test_benchmark_cost_claim_rejects_missing_provider_metadata() -> None:
    with pytest.raises(ValueError, match="provider_metadata"):
        reject_unauditable_cost_claims({"cost_claims": [{"claim": "memory layer cost"}]})


def test_benchmark_cost_claim_rejects_missing_revision_or_cost() -> None:
    payload = {
        "cost_claims": [{"claim": "memory layer cost"}],
        "provider_metadata": [
            {
                "provider_id": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "model_revision": None,
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "latency_ms": 20,
                "monetary_cost_usd": None,
            }
        ],
    }

    with pytest.raises(ValueError, match="model_revision"):
        reject_unauditable_cost_claims(payload)

    payload["provider_metadata"][0]["model_revision"] = "rev"
    with pytest.raises(ValueError, match="monetary_cost_usd"):
        reject_unauditable_cost_claims(payload)


def test_provider_metadata_rejects_invalid_identity_cost_and_token_totals() -> None:
    valid = {
        "provider_id": "openrouter",
        "model_id": "openai/gpt-4o-mini",
        "model_revision": "rev",
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "latency_ms": 20,
        "monetary_cost_usd": 0.001,
    }

    for field in ("provider_id", "model_id"):
        bad = dict(valid)
        bad[field] = None
        with pytest.raises(ValueError, match="non-empty string"):
            reject_unauditable_cost_claims(
                {"cost_claims": [{"claim": "cost"}], "provider_metadata": [bad]}
            )

    with pytest.raises(ValueError, match="provider_id"):
        ProviderMetadata(provider_id=123, model_id="model", model_revision="rev")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_id"):
        ProviderMetadata(provider_id="provider", model_id=True, model_revision="rev")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="provider_id"):
        ProviderMetadata(provider_id=" ", model_id="model", model_revision="rev")
    with pytest.raises(ValueError, match="model_id"):
        ProviderMetadata(provider_id="provider", model_id="", model_revision="rev")
    with pytest.raises(ValueError, match="prompt_tokens"):
        ProviderMetadata(
            provider_id="provider",
            model_id="model",
            model_revision="rev",
            prompt_tokens=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="latency_ms"):
        ProviderMetadata(
            provider_id="provider",
            model_id="model",
            model_revision="rev",
            latency_ms=1.5,  # type: ignore[arg-type]
        )

    bad = dict(valid)
    bad["monetary_cost_usd"] = "NaN"
    with pytest.raises(ValueError, match="finite"):
        reject_unauditable_cost_claims(
            {"cost_claims": [{"claim": "cost"}], "provider_metadata": [bad]}
        )

    bad = dict(valid)
    bad["total_tokens"] = 12
    with pytest.raises(ValueError, match="total_tokens"):
        reject_unauditable_cost_claims(
            {"cost_claims": [{"claim": "cost"}], "provider_metadata": [bad]}
        )


def test_benchmark_artifact_must_declare_cost_claims_even_when_empty() -> None:
    """An absent key is an undeclared posture, not a declaration of no monetary claim."""

    with pytest.raises(ValueError, match="cost_claims"):
        reject_unauditable_cost_claims({"provider_metadata": []})


def test_benchmark_cost_claims_must_be_an_array() -> None:
    with pytest.raises(ValueError, match="cost_claims"):
        reject_unauditable_cost_claims({"cost_claims": "$7.29 per run"})


def test_monetary_prose_is_audited_even_when_cost_claims_is_empty() -> None:
    """The hole this closes: a dollar figure in prose beside an empty `cost_claims` list."""

    with pytest.raises(ValueError, match="provider_metadata"):
        reject_unauditable_cost_claims(
            {"cost_claims": [], "headline": "full LOCOMO arm for $7.29 of tokens"}
        )


def test_monetary_prose_requires_revision_and_cost_fields() -> None:
    payload: dict[str, object] = {
        "cost_claims": [],
        "summary": {"note": "judge spend was 0.75 USD"},
        "provider_metadata": [
            {
                "provider_id": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "model_revision": None,
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "latency_ms": 20,
                "monetary_cost_usd": None,
            }
        ],
    }

    with pytest.raises(ValueError, match="model_revision"):
        reject_unauditable_cost_claims(payload)

    payload["provider_metadata"][0]["model_revision"] = "rev"  # type: ignore[index]
    with pytest.raises(ValueError, match="monetary_cost_usd"):
        reject_unauditable_cost_claims(payload)


def test_monetary_prose_scan_does_not_over_reject_ordinary_artifacts() -> None:
    """Digits in model ids, the word cost without a figure, and verbatim source text all pass."""

    reject_unauditable_cost_claims(
        {
            "cost_claims": [],
            "arm": "recall",
            "model": "openai/gpt-4o-mini",
            "config": {"k": 5, "notes": "cost per question was not measured on this run"},
            "aggregate": {"accuracy": 0.42, "usd": None},
            "outcomes": [
                {"question": "how much did the jacket cost?", "gold": "it was $40"},
            ],
            "provider_metadata": [],
        }
    )


def test_verbatim_source_exclusion_applies_only_at_the_top_level() -> None:
    """A nested key named `outcomes` must not hide a cost claim.

    The exclusion exists for the one top level array of copied in LOCOMO source text. Applied at
    every depth it becomes an audit bypass, and `config["system"]` is `describe()` output from a
    duck typed adapter, so that key namespace is not ours to trust.
    """

    with pytest.raises(ValueError, match="provider_metadata"):
        reject_unauditable_cost_claims(
            {"cost_claims": [], "config": {"system": {"outcomes": "billed $7.29 per run"}}}
        )


def test_monetary_prose_is_found_inside_sets() -> None:
    with pytest.raises(ValueError, match="provider_metadata"):
        reject_unauditable_cost_claims({"cost_claims": [], "tags": {"$7.29 per run"}})


def test_monetary_prose_covers_the_other_currencies_this_project_could_publish() -> None:
    for prose in ("spend was EUR 6.60", "cost €6.60 per run", "£5.00 of tokens"):
        with pytest.raises(ValueError, match="provider_metadata"):
            reject_unauditable_cost_claims({"cost_claims": [], "headline": prose})


def test_pounds_as_a_unit_of_weight_is_not_a_cost_claim() -> None:
    """A false positive here aborts a completed paid run at the write site, so the bare word
    `pounds` is deliberately not a currency form. The £ symbol and `GBP` still are."""

    reject_unauditable_cost_claims(
        {"cost_claims": [], "provider_metadata": [], "aggregate": {"note": "the box weighs 5 pounds"}}
    )


def test_self_referential_payload_does_not_blow_the_stack() -> None:
    """The walk runs before `json.dumps`, so a cycle must not surface as a RecursionError."""

    payload: dict[str, object] = {"cost_claims": []}
    payload["self"] = payload

    reject_unauditable_cost_claims(payload)


def test_write_site_calls_the_validator_before_writing() -> None:
    """A validator the write path stopped calling is a validator that cannot fail."""

    import inspect

    from benchmarks import run as run_module

    source = inspect.getsource(run_module.main)

    assert "reject_unauditable_cost_claims(payload)" in source
    assert source.index("reject_unauditable_cost_claims(payload)") < source.index(
        "path.write_text"
    )


def test_benchmark_without_monetary_claim_may_report_token_metadata_only() -> None:
    reject_unauditable_cost_claims(
        {
            "cost_claims": [],
            "provider_metadata": [
                {
                    "provider_id": "openrouter",
                    "model_id": "openai/gpt-4o-mini",
                    "model_revision": None,
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                    "latency_ms": 20,
                    "monetary_cost_usd": None,
                }
            ],
        }
    )
