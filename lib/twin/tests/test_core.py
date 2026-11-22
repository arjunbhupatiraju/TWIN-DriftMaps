"""Fast checks of TWIN's core claims (pytest tests/)."""
import numpy as np
from twin import DriftModel, BlockTask, PseudoSessions, plant
from twin.tests import p_shuffle, p_pseudo, p_linear_shift


def _rates(test, reps=40, r=0.0, seed=0):
    rng = np.random.default_rng(seed)
    null = PseudoSessions(BlockTask(), "prior")
    dm = DriftModel(slow_frac=0.4, phi=0.97, shared=0.4)
    hits = 0
    for _ in range(reps):
        X = dm.simulate(500, 20, rng); y = null.draw(500, rng)
        if r:
            X = plant(X, y, r, 0.5, rng)
        hits += test(X, y, rng, null) < 0.05
    return hits / reps


def test_shuffle_fails_under_drift():
    assert _rates(lambda X, y, rng, n: p_shuffle(X, y, rng, 100)) > 0.7


def test_valid_tests_hold_alpha():
    assert _rates(lambda X, y, rng, n: p_pseudo(X, y, rng, n, 100)) < 0.15
    assert _rates(lambda X, y, rng, n: p_linear_shift(X, y)) < 0.15


def test_valid_tests_detect_real_code():
    assert _rates(lambda X, y, rng, n: p_linear_shift(X, y), r=0.4) > 0.5


def test_drift_fit_recovers_slow_share():
    rng = np.random.default_rng(1)
    X = DriftModel(slow_frac=0.4, phi=0.97, shared=0.5).simulate(1500, 40, rng)
    assert abs(DriftModel.fit(X).slow_frac - 0.4) < 0.15
