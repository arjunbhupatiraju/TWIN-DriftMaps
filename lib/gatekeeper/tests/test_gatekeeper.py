import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper import characterize, lapse, leadlag  # noqa: E402
from gatekeeper.data import Session  # noqa: E402
from gatekeeper.decision import lapse_verdict, lead_verdict  # noqa: E402
from gatekeeper.glmhmm import GLMHMM  # noqa: E402
from gatekeeper.ibl import _trials, from_cache  # noqa: E402
from gatekeeper.simulate import SimConfig, simulate  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    out = {}
    for sc in ("ni_leads", "null"):
        S = simulate(SimConfig(scenario=sc, n_sessions=6, n_trials=500, seed=7))
        m = GLMHMM(K=2, n_restarts=2, seed=0).fit(S)
        out[sc] = (S, m)
    return out


def test_simulator_deterministic():
    a = simulate(SimConfig(n_sessions=2, n_trials=100, seed=3))
    b = simulate(SimConfig(n_sessions=2, n_trials=100, seed=3))
    assert all((x.choice == y.choice).all() and (x.signals["NI"] == y.signals["NI"]).all() for x, y in zip(a, b))


def test_design_is_causal():
    s = simulate(SimConfig(n_sessions=1, n_trials=50, seed=1))[0]
    X = s.design()
    s2 = Session(s.sid, s.contrast.copy(), s.choice.copy(), {})
    s2.choice[30:] = 1 - np.clip(s2.choice[30:], 0, 1)
    assert np.allclose(X[:31], s2.design()[:31])      # row t uses choices < t only


def test_glmhmm_recovers_states(fitted):
    S, m = fitted["ni_leads"]
    z = np.concatenate([s.truth["state"] for s in S])
    p = np.concatenate([m.p_engaged(s) for s in S])
    assert ((p > 0.5) == z).mean() > 0.85
    assert m.W[0, 0] > 3 and abs(m.W[1, 0]) < 1.5     # engaged state has the steep psychometric
    assert m.A[0, 0] > 0.9 and m.A[1, 1] > 0.9


def test_predicted_posterior_is_causal(fitted):
    S, m = fitted["ni_leads"]
    s = S[0]
    p1 = m.posteriors(s)["predicted"]
    s2 = Session(s.sid, s.contrast.copy(), s.choice.copy(), {})
    s2.choice[200:] = 1 - np.clip(s2.choice[200:], 0, 1)
    p2 = m.posteriors(s2)["predicted"]
    assert np.allclose(p1[:201], p2[:201])


def test_find_switches_stable():
    p = np.r_[np.ones(40), np.zeros(40), np.ones(3), np.zeros(40), np.ones(40)]
    sw = leadlag.find_switches(p, stable=8, win=15)
    assert [t for t, _ in sw] == [40, 123]             # the 3-trial blip is not a switch


def test_onset_recovers_shift():
    k = np.arange(-15, 16)
    for k0 in (-5, 0, 4):
        assert abs(leadlag.onset(1 / (1 + np.exp(-(k - k0) / 1.5))) - k0) < 0.3


def test_leadlag_detects_leader_and_not_null(fitted):
    regs = ["NI", "LC", "pupil"]
    res = {}
    for sc, (S, m) in fitted.items():
        tr = [leadlag.session_traces(s, m.p_engaged(s), regs) for s in S]
        res[sc] = lead_verdict(leadlag.group_stats(tr, regs, n_boot=200))["verdict"]
    assert res == {"ni_leads": "SUPPORTED", "null": "NOT SUPPORTED"}


def test_lapse_forecast_specific(fitted):
    regs = ["NI", "LC", "pupil"]
    out = {}
    for sc, (S, m) in fitted.items():
        lp = lapse.evaluate(lapse.build_table(S, m, regs, horizon=5), regs)
        d = {"NI_beyond_LC_pupil": lapse.session_bootstrap_delta(lp, "behavior+all", "behavior+all-NI", n_boot=200)}
        out[sc] = lapse_verdict(d)["verdict"]
    assert out["null"] == "NOT SUPPORTED"


def test_lapse_horizon_alignment():
    s = simulate(SimConfig(n_sessions=1, n_trials=120, seed=2))[0]

    class M:
        def posteriors(self, s):
            return {"predicted": np.tile([0.5, 0.5], (s.T, 1))}
    row = lapse.build_table([s], M(), ["NI"], horizon=5)[0]
    easy, y = lapse.lapse_labels(s)
    assert (row["y"] == y[5:]).all() and len(row["B"]) == s.T - 5 and len(row["R"]["NI"]) == s.T - 5


def test_characterize_finds_modulated_units():
    S = simulate(SimConfig(scenario="null", n_sessions=3, n_trials=500, seed=4))
    rows = characterize.unit_modulation(S, [s.truth["state"].astype(float) for s in S], "NI", n_perm=200)
    assert characterize.summarize(rows)["frac_significant"] > 0.3
    S = simulate(SimConfig(scenario="noise", n_sessions=3, n_trials=500, seed=4))
    rows = characterize.unit_modulation(S, [s.truth["state"].astype(float) for s in S], "NI", n_perm=200)
    assert characterize.summarize(rows)["frac_significant"] < 0.15


class FakeOne:
    def __init__(self, right_val):
        rng = np.random.default_rng(0)
        n = 200
        c = rng.choice([0, .25, 1], n) * rng.choice([-1, 1], n)
        right = (c > 0) | ((c == 0) & (rng.random(n) < .5))
        ch = np.where(right, right_val, -right_val)
        ch[:5] = 0
        self.t = dict(contrastLeft=np.where(c < 0, -c, np.nan), contrastRight=np.where(c > 0, c, np.nan),
                      choice=ch, feedbackType=np.where((right == (c > 0)) | (c == 0), 1, -1),
                      stimOn_times=np.arange(n) * 3.0, probabilityLeft=np.full(n, .5))
        self.c = c

    def load_object(self, eid, obj):
        return self.t


@pytest.mark.parametrize("right_val", [-1, 1])
def test_ibl_choice_mapping(right_val):
    one = FakeOne(right_val)
    contrast, choice, onsets, meta = _trials(one, "x")
    assert meta["right_value"] == right_val
    nz = contrast != 0
    assert ((choice[nz][5:] == 1) == (contrast[nz][5:] > 0)).all()
    assert (choice[:5] == -1).all()


def test_cache_roundtrip(tmp_path):
    p = tmp_path / "abc.npz"
    np.savez_compressed(p, contrast=np.array([.5, -1.]), choice=np.array([1, 0]), meta=json.dumps({"units": {"NI": 2}}),
                        sig__NI=np.ones((2, 2)), sig__pupil=np.ones((2, 1)))
    s = from_cache(p)
    assert s.sid == "abc" and set(s.signals) == {"NI", "pupil"} and s.meta["units"]["NI"] == 2


class FakeDatasets:
    def __init__(self, files):
        self.files = files

    def load_dataset(self, eid, name, collection=None):
        if name not in self.files:
            raise KeyError(name)
        return self.files[name]


def test_region_labels_fallbacks():
    from gatekeeper.ibl import region_labels
    import pandas as pd
    a = region_labels(FakeDatasets({"clusters.brainLocationAcronyms_ccf_2017.npy": np.array(["NI", "LC"])}), "e", "c")
    b = region_labels(FakeDatasets({"clusters.brainLocationIds_ccf_2017.npy": np.array([604, 147]),
                                    "clusters.metrics.pqt": pd.DataFrame({"label": [1.0, 0.33]})}), "e", "c")
    c = region_labels(FakeDatasets({"clusters.channels.npy": np.array([1, 0]),
                                    "channels.brainLocationIds_ccf_2017.npy": np.array([147, 604])}), "e", "c")
    assert list(a[0]) == list(b[0]) == list(c[0]) == ["NI", "LC"]
    assert list(b[1]) == [1.0, 0.33] and np.isnan(a[1]).all()
    assert region_labels(FakeDatasets({}), "e", "c") is None


# ---------------- brain-wide ----------------
def test_bh_matches_definition():
    from gatekeeper.brainwide import bh
    p = np.array([0.01, 0.04, 0.03, 0.2])
    q = bh(p)
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.2])   # hand-computed BH


def test_brainwide_sim_lead_direction():
    from gatekeeper.simulate import BrainwideConfig, simulate_brainwide
    regs = {"L": dict(kind="lead", lead=5, units=30, p_present=1.0)}
    s = simulate_brainwide(BrainwideConfig(n_sessions=1, n_trials=3000, regions=regs, seed=1))[0]
    x = s.signals["L"]
    w = np.linalg.lstsq(x - x.mean(0), s.truth["state"] - s.truth["state"].mean(), rcond=None)[0]
    proj = (x - x.mean(0)) @ w
    z = s.truth["state"].astype(float)
    cc = [np.corrcoef(proj[:-k], z[k:])[0, 1] if k else np.corrcoef(proj, z)[0, 1] for k in range(0, 8)]
    assert np.argmax(cc) >= 2                  # region best predicts the state several trials later


def test_heterogeneity():
    from gatekeeper.brainwide import heterogeneity
    same = [dict(onset=0.1 * i, onset_se=1.0) for i in range(5)]
    diff = [dict(onset=-6.0, onset_se=0.5)] + [dict(onset=0.0, onset_se=0.5) for _ in range(4)]
    assert heterogeneity(same)["p"] > 0.5 and heterogeneity(diff)["p"] < 1e-6


def test_counts_fast_matches_slow():
    from gatekeeper.ibl import _counts, counts_fast
    rng = np.random.default_rng(0)
    st = np.sort(rng.uniform(0, 500, 50000))
    sc = rng.integers(0, 30, 50000)
    on = np.sort(rng.uniform(1, 499, 200))
    ids = np.array([0, 5, 29, 12])
    assert np.array_equal(counts_fast(st, sc, ids, on), _counts(st, sc, ids, on))


def test_brainwide_finds_leader_small():
    from gatekeeper import brainwide
    from gatekeeper.simulate import BrainwideConfig, simulate_brainwide
    regs = {"LEAD": dict(kind="lead", lead=4, units=10, p_present=0.8),
            "STATE": dict(kind="state", lead=0, units=10, p_present=0.8),
            "NOISE": dict(kind="none", lead=0, units=10, p_present=0.8)}
    S = simulate_brainwide(BrainwideConfig(n_sessions=10, n_trials=500, regions=regs, seed=4))
    m = GLMHMM(K=2, n_restarts=1, seed=0).fit(S)
    res = brainwide.analyze(S, m, sorted(regs), n_boot=150, log=lambda *a: None)
    assert brainwide.verdict(res)["H1_some_region_leads"]["leaders"] == ["LEAD"]


# ---------------- INFOFLOW ----------------
def test_cv_dprime_null_and_signal():
    from gatekeeper.infoflow import cv_dprime
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    X0 = rng.normal(0, 1, (400, 10))
    X1 = X0 + y[:, None] * 0.5
    assert abs(cv_dprime(X0, y)) < 0.3 and cv_dprime(X1, y) > 1.0


def test_loo_posterior_ignores_own_choice():
    S = simulate(SimConfig(n_sessions=2, n_trials=300, seed=5))
    m = GLMHMM(K=2, n_restarts=1, seed=0).fit(S)
    s = S[0]
    a = m.posteriors(s)["loo"]
    s2 = Session(s.sid, s.contrast.copy(), s.choice.copy(), {})
    s2.choice[150] = 1 - max(s2.choice[150], 0)
    b = m.posteriors(s2)
    # trial 150's own label is unchanged; neighbours may change slightly but not trial 150 itself
    assert np.allclose(a[150], b["loo"][150])
    assert not np.allclose(m.posteriors(s)["smoothed"][150], b["smoothed"][150])


def test_split_is_deterministic():
    from gatekeeper.infoflow_data import split_of
    assert split_of("SWC_001") == split_of("SWC_001")
    assert 0.3 < np.mean([split_of(f"m{i}") == "explore" for i in range(1000)]) < 0.5


def test_lapse_locus_on_synthetic_rows():
    from gatekeeper.infoflow import lapse_locus
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(12):
        for st, rho in (("1 visual thalamus", 1.0), ("2 primary visual cortex", 1.0), ("4 midbrain", 0.3)):
            de = 1.0 + rng.normal(0, 0.05)
            rows.append(dict(sid=f"s{sid}", stage=st, d_engaged_all=de, d_disengaged_all=de * rho + rng.normal(0, 0.05),
                             db_engaged=1.5, db_disengaged=0.3))
    L = lapse_locus(rows, n_boot=300)
    assert L["locus"] == "4 midbrain" and not L["stages"]["1 visual thalamus"]["lossy"]
