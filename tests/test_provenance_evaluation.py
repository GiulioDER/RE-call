from benchmarks.provenance_controller_eval import run_evaluation


def test_preregistered_provenance_evaluation_fixture_is_safe_and_complete() -> None:
    artifact = run_evaluation()
    metrics = artifact["metrics"]

    assert artifact["fixture"] == "preregistered-provenance-controller-v1"
    assert artifact["case_count"] == 15
    assert metrics["unauthorized_stale_application_rate"] == 0.0
    assert metrics["unauthorized_contradictory_application_rate"] == 0.0
    assert metrics["trusted_present_evidence_acceptance_rate"] == 1.0
    assert metrics["false_abstention_rate_supported_current"] == 0.0
    assert metrics["fresh_search_recovery_rate"] == 0.5
    assert metrics["duplicate_application_rate"] == 1.0
    assert metrics["refusal_counts"]["LEDGER_UNAVAILABLE"] == 1
    assert metrics["refusal_counts"]["MATERIALIZATION_UNAVAILABLE"] == 1

    for row in artifact["cases"]:
        if row["case_id"] == "materializer_outage":
            continue
        if not row["allowed"]:
            assert row["asserted_events_delta"] == 0
