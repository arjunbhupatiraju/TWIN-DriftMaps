"""Trial-level session container shared by the simulator and the IBL loader."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Session:
    sid: str
    contrast: np.ndarray          # (T,) signed contrast, + = right, in [-1, 1]
    choice: np.ndarray            # (T,) 1 = right, 0 = left, -1 = no-go / missing
    signals: dict                 # region -> (T, N) per-trial pre-stimulus activity (spike counts or pupil)
    meta: dict = field(default_factory=dict)
    truth: dict = field(default_factory=dict)   # simulator only: hidden state etc. Never used by analyses.
    block: np.ndarray = None      # optional: IBL block prior, +1 = right-likely block, -1 = left-likely, 0 = unbiased

    @property
    def T(self):
        return len(self.contrast)

    @property
    def valid(self):
        return self.choice >= 0

    @property
    def correct(self):
        side = np.where(self.contrast > 0, 1, np.where(self.contrast < 0, 0, -1))
        return np.where(side < 0, -1, (self.choice == side).astype(int))

    def design(self):
        """GLM-HMM inputs (Ashwood et al. 2022 style): stimulus, bias, previous choice, win-stay-lose-switch."""
        c = self.contrast
        stim = np.tanh(5 * c) / np.tanh(5)                 # compresses high contrasts as in Ashwood 2022
        prev = np.r_[0.0, 2 * self.choice[:-1] - 1.0]
        prev[np.r_[True, self.choice[:-1] < 0]] = 0.0
        corr = self.correct
        prev_rew = np.r_[0.0, np.where(corr[:-1] == 1, 1.0, np.where(corr[:-1] == 0, -1.0, 0.0))]
        wsls = prev * prev_rew
        X = np.c_[stim, np.ones_like(stim), prev, wsls]
        if self.block is not None:
            # IBL biased blocks: without this input a 2-state GLM-HMM splits 'leaning left' vs
            # 'leaning right' (found on real data) instead of engaged vs disengaged.
            X = np.c_[X, np.asarray(self.block, float)]
        return X
