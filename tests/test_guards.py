from datetime import datetime, timedelta, timezone

from recall.guards import gap_warning, staleness


def test_gap_warning_true_when_all_scores_below_threshold():
    assert gap_warning([0.10, 0.22, 0.30]) is True


def test_gap_warning_false_when_any_score_meets_threshold():
    assert gap_warning([0.10, 0.61, 0.30]) is False


def test_gap_warning_true_on_empty():
    assert gap_warning([]) is True


def test_gap_warning_respects_custom_threshold():
    assert gap_warning([0.4], threshold=0.3) is False


def test_staleness_stale_when_older_than_max_age():
    now = datetime(2026, 7, 6, tzinfo=timezone.utc)
    old = now - timedelta(days=5)
    r = staleness(old, now, timedelta(days=2))
    assert r.stale is True
    assert r.age == timedelta(days=5)


def test_staleness_fresh_when_within_max_age():
    now = datetime(2026, 7, 6, tzinfo=timezone.utc)
    recent = now - timedelta(hours=1)
    r = staleness(recent, now, timedelta(days=2))
    assert r.stale is False


def test_staleness_stale_when_never_indexed():
    now = datetime(2026, 7, 6, tzinfo=timezone.utc)
    r = staleness(None, now, timedelta(days=2))
    assert r.stale is True
    assert r.newest_indexed_at is None
