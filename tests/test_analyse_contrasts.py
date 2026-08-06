from benchmarks.mtrag.analyse_contrasts import holm, paired_stats
import random

def test_holm_step_down_stops_at_the_first_failure():
    """A hypothesis weaker than a failed one cannot pass, even if its own p clears alpha."""
    # c's p (0.030) EXCEEDS its step-down threshold 0.05/2 = 0.025, so it fails and latches.
    # b's p (0.040) would clear the final threshold 0.05/1 = 0.05 on its own, and must not.
    r = {"a": {"p_permutation": 0.001}, "b": {"p_permutation": 0.040}, "c": {"p_permutation": 0.030}}
    holm(r, alpha=0.05)
    assert r["a"]["holm_significant"] is True    # 0.001 < 0.05/3 = 0.0167
    assert r["c"]["holm_significant"] is False   # 0.030 > 0.025 -> fails, latches the chain
    assert r["b"]["holm_significant"] is False   # 0.040 < 0.05 alone, but stopped by c

def test_holm_thresholds_step_down():
    r = {"a": {"p_permutation": 0.001}, "b": {"p_permutation": 0.002}}
    holm(r, alpha=0.05)
    assert r["a"]["holm_threshold"] == 0.025
    assert r["b"]["holm_threshold"] == 0.05

def test_a_zero_effect_gives_a_ci_containing_zero():
    """Sanity on the apparatus itself: identical arms must not look significant."""
    res = paired_stats([0.0] * 200, random.Random(0))
    assert res["mean_delta"] == 0.0
    assert not res["ci_excludes_zero"]
    assert res["n_nonzero"] == 0

def test_a_constant_positive_effect_is_detected():
    res = paired_stats([0.1] * 200, random.Random(0))
    assert res["ci_low"] > 0 and res["ci_excludes_zero"]
    assert res["p_permutation"] < 0.01
