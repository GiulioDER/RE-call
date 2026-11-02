import pytest

from benchmarks.mtrag import buried_gold_power
from benchmarks.mtrag.buried_gold_power import mcnemar_power, minimum_detectable_shift


def test_no_effect_gives_power_near_alpha():
    """With treatment == control the test should reject at roughly its nominal rate. This is the
    sanity check that the simulation is a test and not a rubber stamp."""
    power = mcnemar_power(n=123, p_control=0.73, p_treatment=0.73, rho=0.5, trials=20000, seed=1)
    assert power < 0.10


def test_large_effect_is_well_powered():
    power = mcnemar_power(n=123, p_control=0.73, p_treatment=0.35, rho=0.5, trials=20000, seed=1)
    assert power > 0.95


def test_power_increases_with_effect_size():
    small = mcnemar_power(n=123, p_control=0.73, p_treatment=0.68, rho=0.5, trials=20000, seed=1)
    large = mcnemar_power(n=123, p_control=0.73, p_treatment=0.50, rho=0.5, trials=20000, seed=1)
    assert large > small


def test_power_increases_with_n():
    small_n = mcnemar_power(n=40, p_control=0.73, p_treatment=0.55, rho=0.5, trials=20000, seed=1)
    large_n = mcnemar_power(n=400, p_control=0.73, p_treatment=0.55, rho=0.5, trials=20000, seed=1)
    assert large_n > small_n


def test_minimum_detectable_shift_is_below_control():
    mds = minimum_detectable_shift(n=123, p_control=0.73, rho=0.5)
    assert mds is not None
    assert 0.0 <= mds < 0.73


def test_minimum_detectable_shift_is_actually_powered(monkeypatch):
    """NUM-001: a6051cac5 fails the marginal assertion; power is checked at its sampled rate."""
    rates = []

    def power_for_rate(*args, **kwargs):
        rates.append(args[2])
        return mcnemar_power(*args, **kwargs)

    monkeypatch.setattr(buried_gold_power, "mcnemar_power", power_for_rate)
    mds = minimum_detectable_shift(n=123, p_control=0.73, rho=0.5)
    assert mds == pytest.approx(round(0.5 * 0.73 + 0.5 * rates[-1], 4))
    power = mcnemar_power(n=123, p_control=0.73, p_treatment=rates[-1], rho=0.5,
                          trials=20000, seed=2)
    assert power >= 0.78


def test_minimum_detectable_shift_reports_the_achieved_marginal(monkeypatch):
    """NUM-001: a6051cac5 returns 0.30, failing the independently derived 0.515 assertion."""
    def controlled_power(n, p_control, p_treatment, rho, alpha, trials, seed):
        return float(p_treatment <= 0.30)

    monkeypatch.setattr(buried_gold_power, "mcnemar_power", controlled_power)
    assert minimum_detectable_shift(123, 0.73, 0.5) == pytest.approx(0.515)
    # With no copied outcomes, the independent and achieved rates coincide.
    assert minimum_detectable_shift(123, 0.73, 0.0) == pytest.approx(0.30)


def test_tiny_n_may_be_unpowered_at_any_effect():
    """The case the spec's demotion rule exists for. `None` means no shift in range is detectable,
    and Family B becomes descriptive with no p-value attached."""
    assert minimum_detectable_shift(n=3, p_control=0.73, rho=0.5, target_power=0.99) is None
