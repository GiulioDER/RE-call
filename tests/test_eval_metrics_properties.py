"""Property checks for invariants shared by retrieval evaluation metrics."""

from hypothesis import HealthCheck, given, settings, strategies as st

from recall.eval.metrics import precision_at_k, recall_at_k

_IDS = st.sampled_from(("a", "b", "c", "d", ""))


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    retrieved=st.lists(_IDS, max_size=40),
    relevant=st.lists(_IDS, max_size=20),
    k=st.integers(min_value=0, max_value=50),
)
def test_precision_and_recall_stay_bounded(
    retrieved: list[str], relevant: list[str], k: int
) -> None:
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    retrieved=st.lists(_IDS, max_size=40),
    relevant=st.lists(_IDS, max_size=20),
    first_k=st.integers(min_value=0, max_value=50),
    second_k=st.integers(min_value=0, max_value=50),
)
def test_recall_is_monotonic_as_the_retrieved_prefix_grows(
    retrieved: list[str],
    relevant: list[str],
    first_k: int,
    second_k: int,
) -> None:
    small_k, large_k = sorted((first_k, second_k))

    assert recall_at_k(retrieved, relevant, small_k) <= recall_at_k(
        retrieved, relevant, large_k
    )
