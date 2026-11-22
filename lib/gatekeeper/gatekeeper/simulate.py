"""Synthetic IBL-like sessions with a KNOWN answer, used to validate every analysis before real data.

A slow latent arousal variable a_t drives switches between an engaged and a disengaged
behavioral state with a delay of `lead` trials:

    logit P(engaged_t) = stick * (2 z_{t-1} - 1) + drive * a_{t-lead}

Regions are built from either the latent (a_t: the region "leads" state switches by
`lead` trials), the state itself (z_t: concurrent readout, no lead), a lagged copy of the
state (follower), or nothing (noise). Scenarios:

    ni_leads   NI <- a_t        LC <- z_t        pupil <- z_{t-2} smoothed
    lc_leads   NI <- z_t        LC <- a_t        pupil <- z_{t-2} smoothed
    null       NI <- z_t        LC <- z_t        pupil <- z_{t-2} smoothed   (nobody leads)
    noise      NI <- nothing    LC <- z_t        pupil <- z_{t-2} smoothed
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Session

CONTRASTS = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])
SCENARIOS = {
    "ni_leads": dict(NI="latent", LC="state", pupil="follow"),
    "lc_leads": dict(NI="state", LC="latent", pupil="follow"),
    "null": dict(NI="state", LC="state", pupil="follow"),
    "noise": dict(NI="none", LC="state", pupil="follow"),
}


@dataclass
class SimConfig:
    scenario: str = "ni_leads"
    n_sessions: int = 12
    n_trials: int = 600
    lead: int = 4                  # trials by which the latent precedes behavioral switches
    phi: float = 0.97              # latent AR(1) coefficient (slow arousal)
    stick: float = 3.0
    drive: float = 3.5
    w_engaged: tuple = (5.0, 0.0, 0.3, 0.2)      # stim, bias, prev choice, wsls
    w_disengaged: tuple = (0.6, 0.0, 0.8, 0.0)   # bias drawn per session
    units: dict = None             # region -> number of units
    gain: float = 0.35             # log-rate modulation per unit of signal (unit loadings ~ gain)
    base_rate: float = 6.0         # spikes per pre-stimulus window
    nogo_disengaged: float = 0.05
    seed: int = 0

    def __post_init__(self):
        if self.units is None:
            self.units = dict(NI=12, LC=12)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    roles = SCENARIOS[cfg.scenario]
    out = []
    for s in range(cfg.n_sessions):
        T, L = cfg.n_trials, cfg.lead
        a = np.zeros(T + L)
        for t in range(1, T + L):
            a[t] = cfg.phi * a[t - 1] + np.sqrt(1 - cfg.phi ** 2) * rng.standard_normal()
        a_now, a_drv = a[L:], a[:T]           # a_now[t] = a_t, a_drv[t] = a_{t-L}
        z = np.zeros(T, int)
        z[0] = 1
        for t in range(1, T):
            z[t] = rng.random() < _sigmoid(cfg.stick * (2 * z[t - 1] - 1) + cfg.drive * a_drv[t])
        contrast = rng.choice(CONTRASTS, T) * rng.choice([-1, 1], T)
        bias_dis = rng.normal(0, 1.5)
        choice = np.zeros(T, int)
        sess = Session(f"sim{s}", contrast, choice, {})
        X_partial = None
        for t in range(T):                    # sequential because inputs depend on past choices
            sess.choice = choice
            X_partial = sess.design()[t] if t else np.array([np.tanh(5 * contrast[0]) / np.tanh(5), 1, 0, 0])
            w = np.array(cfg.w_engaged if z[t] else cfg.w_disengaged, float)
            if not z[t]:
                w[1] = bias_dis
            choice[t] = rng.random() < _sigmoid(X_partial @ w)
            if not z[t] and rng.random() < cfg.nogo_disengaged:
                choice[t] = -1
        zc = z - z.mean()
        follow = np.convolve(np.r_[np.zeros(2), zc[:-2]], np.ones(3) / 3, "same")
        sources = dict(latent=a_now, state=zc * 1.6, follow=follow * 1.6, none=np.zeros(T))
        signals = {}
        for region, n in cfg.units.items():
            x = sources[roles[region]]
            load = rng.normal(0, 1, n) * cfg.gain
            lam = cfg.base_rate * np.exp(np.outer(x, load) + 0.3 * rng.standard_normal((T, n)))
            signals[region] = rng.poisson(lam).astype(float)
        p = sources[roles["pupil"]] + 0.5 * rng.standard_normal(T)
        signals["pupil"] = p[:, None]
        out.append(Session(f"{cfg.scenario}_s{s}", contrast, choice, signals,
                           meta=dict(scenario=cfg.scenario, lead=L),
                           truth=dict(state=z, latent=a_now, roles=roles)))
    return out


# ---------------------------------------------------------------------------
# Brain-wide simulator: many regions, each with a known role and lead time.
# ---------------------------------------------------------------------------
@dataclass
class BrainwideConfig:
    n_sessions: int = 30
    n_trials: int = 600
    max_lead: int = 8              # behavior is driven by the latent max_lead trials back
    regions: dict = None           # name -> dict(kind, lead, units, p_present)
    phi: float = 0.97
    stick: float = 3.0
    drive: float = 3.5
    w_engaged: tuple = (5.0, 0.0, 0.3, 0.2)
    w_disengaged: tuple = (0.6, 0.0, 0.8, 0.0)
    gain: float = 0.35
    base_rate: float = 6.0
    seed: int = 0

    def __post_init__(self):
        if self.regions is None:
            self.regions = default_brainwide_regions()


def default_brainwide_regions():
    """Ground truth used for validation. kind:
    lead    region tracks the latent `lead` trials before behavior switches (true leader)
    state   concurrent readout of the behavioral state (no lead)
    follow  lags the state by `lead` trials (smoothed)
    none    no engagement signal
    """
    R = {}
    for i, k in enumerate((6, 4, 2)):
        R[f"LEAD{k}"] = dict(kind="lead", lead=k, units=8, p_present=0.4)
    for i in range(6):
        R[f"STATE{i}"] = dict(kind="state", lead=0, units=8, p_present=0.4)
    for i, k in enumerate((2, 4)):
        R[f"FOLLOW{k}"] = dict(kind="follow", lead=k, units=8, p_present=0.4)
    for i in range(8):
        R[f"NOISE{i}"] = dict(kind="none", lead=0, units=8, p_present=0.4)
    return R


def global_null_regions():
    """No region leads: half concurrent, some followers, rest noise."""
    R = {f"STATE{i}": dict(kind="state", lead=0, units=8, p_present=0.4) for i in range(8)}
    R.update({f"FOLLOW{k}": dict(kind="follow", lead=k, units=8, p_present=0.4) for k in (2, 4)})
    R.update({f"NOISE{i}": dict(kind="none", lead=0, units=8, p_present=0.4) for i in range(8)})
    return R


def simulate_brainwide(cfg: BrainwideConfig):
    rng = np.random.default_rng(cfg.seed)
    out = []
    names = sorted(cfg.regions)
    for s in range(cfg.n_sessions):
        T, L = cfg.n_trials, cfg.max_lead
        a = np.zeros(T + L)
        for t in range(1, T + L):
            a[t] = cfg.phi * a[t - 1] + np.sqrt(1 - cfg.phi ** 2) * rng.standard_normal()
        z = np.zeros(T, int)
        z[0] = 1
        for t in range(1, T):
            z[t] = rng.random() < _sigmoid(cfg.stick * (2 * z[t - 1] - 1) + cfg.drive * a[t])   # a[t] = latent_{t-L}
        contrast = rng.choice(CONTRASTS, T) * rng.choice([-1, 1], T)
        bias_dis = rng.normal(0, 1.5)
        choice = np.zeros(T, int)
        sess = Session("tmp", contrast, choice, {})
        for t in range(T):
            sess.choice = choice
            x = sess.design()[t] if t else np.array([np.tanh(5 * contrast[0]) / np.tanh(5), 1, 0, 0])
            w = np.array(cfg.w_engaged if z[t] else cfg.w_disengaged, float)
            if not z[t]:
                w[1] = bias_dis
            choice[t] = rng.random() < _sigmoid(x @ w)
        zc = (z - z.mean()) * 1.6
        signals = {}
        present = [r for r in names if rng.random() < cfg.regions[r]["p_present"]]
        for r in present:
            spec = cfg.regions[r]
            k = spec["lead"]
            if spec["kind"] == "lead":
                x = a[k: k + T]                # x_t = a[t+k], which drives behavior at trial t+k: leads by k
            elif spec["kind"] == "state":
                x = zc
            elif spec["kind"] == "follow":
                x = np.convolve(np.r_[np.zeros(k), zc[:-k]], np.ones(3) / 3, "same")
            else:
                x = np.zeros(T)
            n = max(1, int(rng.poisson(spec["units"])))
            load = rng.normal(0, 1, n) * cfg.gain
            lam = cfg.base_rate * np.exp(np.outer(x, load) + 0.3 * rng.standard_normal((T, n)))
            signals[r] = rng.poisson(lam).astype(float)
        follow2 = np.convolve(np.r_[np.zeros(2), zc[:-2]], np.ones(3) / 3, "same")
        signals["pupil"] = (follow2 + 0.5 * rng.standard_normal(T))[:, None]
        out.append(Session(f"bw_s{s}", contrast, choice, signals,
                           truth=dict(state=z, regions={r: cfg.regions[r] for r in present})))
    return out
