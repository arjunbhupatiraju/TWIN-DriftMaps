import numpy as np
import pytest

from driftaware import bh, drift_test, fisher, imposter_labels, region_map, shuffle_labels, simulate_session
from driftaware.simulate import block_labels


def _rates(beta, n=60, method="drift", null="pseudo"):
    out = []
    for i in range(n):
        X, y = simulate_session(T=400, N=20, beta=beta, seed=1000 + i)
        nul = (lambda r: block_labels(400, r)) if null == "pseudo" else shuffle_labels(y)
        out.append(drift_test(X, y, nul, n_null=60, seed=i, method=method).p < 0.05)
    return np.mean(out)


def test_calibrated_under_drift():
    assert _rates(0.0) <= 0.12


def test_shuffle_null_is_inflated():
    assert _rates(0.0, method="plain", null="shuffle") >= 0.5


def test_power_beats_plain():
    assert _rates(0.08, n=40) > _rates(0.08, n=40, method="plain")


def test_imposter_generator_shapes():
    pool = [np.arange(50.0), np.arange(30.0)]
    draw = imposter_labels(pool, 120)
    y = draw(np.random.default_rng(0))
    assert y.shape == (120,)


def test_array_nulls_and_validation():
    X, y = simulate_session(T=200, N=10, seed=3)
    Y = np.column_stack([np.random.default_rng(i).permutation(y) for i in range(20)])
    r = drift_test(X, y, Y)
    assert 0 < r.p <= 1 and len(r.null) == 20
    with pytest.raises(ValueError):
        drift_test(X[:-1], y, Y)


def test_maps():
    assert fisher([1.0, 1.0]) == pytest.approx(1.0)
    assert bh([0.001, 0.5, 0.9]).tolist() == [True, False, False]
    import pandas as pd
    df = pd.DataFrame({"region": ["A", "A", "B"], "p": [0.001, 0.002, 0.8]})
    m = region_map(df, "p")
    assert bool(m.loc["A", "significant"]) and not bool(m.loc["B", "significant"])
