"""Simulated 7-stage visual-to-motor pathway with a KNOWN locus of lapse-related information loss.

Stage 1 encodes the signed stimulus with information-limiting (differential) correlations, so its
information has a finite ceiling. Each later stage is a noisy random re-mapping of the previous one
(information can only fall: data-processing inequality). The choice is read out from stage 7.
A sticky 2-state Markov chain sets engagement. In DISENGAGED trials:
    source      stage-1 stimulus gain is reduced (information lost at the source, so everywhere)
    downstream  extra signal-direction noise is injected at stage 4 (stages 1-3 intact)
    decision    neural information intact everywhere; only the final readout gets noisier
Each session records a random subset of stages, 8-40 neurons per recorded stage (Poisson counts).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Session

STAGE_REGIONS = ["LGd", "VISp", "VISa", "SCm", "CP", "MOs", "GRN"]
CONTRASTS = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])
SCENARIOS = ("source", "downstream", "decision")


@dataclass
class PathwayConfig:
    scenario: str = "downstream"
    n_sessions: int = 40
    n_trials: int = 600
    pop: int = 120                # true population size per stage
    eps: float = 0.15             # differential-correlation strength at stage 1 (ceiling ~ 1/eps)
    stage_noise: float = 0.6      # independent noise added at each re-mapping
    lapse_factor: float = 0.25    # disengaged gain (source) / signal-noise multiplier
    p_present: float = 0.5
    seed: int = 0


class SimInfoSession:
    def __init__(self, sid, contrast, choice, rt, arrays, split="explore", truth=None):
        self.sid, self.contrast, self.choice, self.rt = sid, contrast, choice, rt
        self._a, self.split, self.truth = arrays, split, truth or {}
        self.meta = dict(subject=sid, split=split)
        self.regions = sorted(arrays)

    def X(self, region, window="early"):
        return self._a.get(region)

    def as_behavior_session(self):
        return Session(self.sid, self.contrast, self.choice, {})


def _ceff(c):
    return np.abs(c) / (np.abs(c) + 0.12) * np.sign(c) * 1.12


def simulate_pathway(cfg: PathwayConfig):
    rng = np.random.default_rng(cfg.seed)
    P, K = cfg.pop, len(STAGE_REGIONS)
    f1 = rng.normal(0, 1, P)
    f1 /= np.linalg.norm(f1)
    Ws = [rng.normal(0, 1 / np.sqrt(P), (P, P)) for _ in range(K - 1)]
    out = []
    for s in range(cfg.n_sessions):
        T = cfg.n_trials
        z = np.zeros(T, int)
        z[0] = 1
        for t in range(1, T):
            z[t] = z[t - 1] if rng.random() < (0.97 if z[t - 1] else 0.94) else 1 - z[t - 1]
        c = rng.choice(CONTRASTS, T) * rng.choice([-1, 1], T)
        sig = _ceff(c)
        dis = z == 0
        gain = np.where(dis & (cfg.scenario == "source"), cfg.lapse_factor, 1.0)
        # stage 1: signal along f1 + independent noise + differential (information-limiting) noise
        r = (gain * sig)[:, None] * f1[None, :] * 3.0 + rng.normal(0, 1, (T, P)) \
            + np.sqrt(cfg.eps) * 3.0 * rng.normal(0, 1, T)[:, None] * f1[None, :]
        stages = [r]
        for k in range(1, K):
            r = np.tanh(stages[-1] @ Ws[k - 1].T) * 2.0 + rng.normal(0, cfg.stage_noise, (T, P))
            if k == 3 and cfg.scenario == "downstream":
                # lapse = stimulus GAIN reduction at stage 4 (each neuron's stimulus-locked response is
                # scaled by lapse_factor on disengaged trials); stages 1-3 intact, 4-7 inherit the loss
                eng = ~dis
                sc = sig - sig[eng].mean()
                b = ((r[eng] - r[eng].mean(0)) * sc[eng][:, None]).sum(0) / (sc[eng] ** 2).sum()
                r = r - (dis * (1 - cfg.lapse_factor) * sc)[:, None] * b[None, :]
            stages.append(r)
        # readout from stage 7: direction that best separates the signal (fit on engaged, noise-free proxy)
        last = stages[-1]
        w = np.linalg.lstsq(last[~dis] - last[~dis].mean(0), sig[~dis], rcond=None)[0]
        dv = (last - last.mean(0)) @ w
        dv = dv / (dv[~dis].std() + 1e-9) * 1.5
        noise = rng.normal(0, 1, T) * np.where(dis & (cfg.scenario == "decision"), 1 / cfg.lapse_factor, 1.0)
        bias = rng.normal(0, 1.0)
        choice = (dv + noise + np.where(dis, bias, 0) > 0).astype(int)
        rt = 0.25 + rng.gamma(2, 0.08, T)
        arrays = {}
        for k, reg in enumerate(STAGE_REGIONS):
            if rng.random() > cfg.p_present:
                continue
            m = int(rng.integers(8, 41))
            idx = rng.choice(P, m, replace=False)
            lam = 4.0 * np.exp(0.35 * np.clip(stages[k][:, idx], -6, 6))
            arrays[reg] = rng.poisson(lam).astype(float)
        out.append(SimInfoSession(f"{cfg.scenario}_s{s}", c, choice, rt, arrays, truth=dict(state=z)))
    return out
