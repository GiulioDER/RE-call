from recall.retrieval_slo import QUALITY_RETRIEVAL_SLO


def test_quality_slo_is_explicitly_measured_and_bounded() -> None:
    assert QUALITY_RETRIEVAL_SLO.profile == "quality"
    assert QUALITY_RETRIEVAL_SLO.p95_ms == 2_000
    assert QUALITY_RETRIEVAL_SLO.p99_ms == 2_200
    assert QUALITY_RETRIEVAL_SLO.max_offered_concurrency == 4
    assert QUALITY_RETRIEVAL_SLO.max_rss_bytes == 1_280 * 1024 * 1024
    assert QUALITY_RETRIEVAL_SLO.max_error_rate == 0.01
    assert QUALITY_RETRIEVAL_SLO.alert_window_minutes == 5
