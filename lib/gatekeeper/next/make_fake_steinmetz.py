"""Synthetic data in the exact Neuromatch Steinmetz format, to test the pipeline end to end (not real data)."""
import sys
from pathlib import Path

import numpy as np


def session(rng, mouse, T=320, areas=("VISp", "MOs", "CA1", "root"), n_per=12):
    cl = rng.choice([0, .25, .5, 1], T)
    cr = rng.choice([0, .25, .5, 1], T)
    resp = np.where(rng.random(T) < .1, 0, np.where(cr > cl, -1, 1) * np.where(rng.random(T) < .8, 1, -1)).astype(float)
    resp[cl == cr] = rng.choice([-1, 1], (cl == cr).sum())
    fb = np.where(rng.random(T) < .7, 1, -1).astype(float)
    rt = rng.uniform(.3, 1.2, T)
    ft = rt + .05
    N = n_per * len(areas)
    slow = np.zeros((N, T))
    z = rng.standard_normal(N)
    for t in range(T):
        z = .98 * z + np.sqrt(1 - .98 ** 2) * rng.standard_normal(N)
        slow[:, t] = z
    rate = np.exp(-1.5 + 0.5 * slow)[:, :, None] * np.ones((1, 1, 250))
    spks = rng.poisson(rate).astype(np.int8)
    return dict(spks=spks, brain_area=np.repeat(areas, n_per), contrast_left=cl, contrast_right=cr, response=resp,
                feedback_type=fb, response_time=rt[:, None], feedback_time=ft[:, None], gocue=np.full((T, 1), .5),
                mouse_name=mouse, date_exp="2017-01-01")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dat = np.array([session(rng, f"M{i % 4}") for i in range(8)], dtype=object)
    np.savez(out / "steinmetz_part0.npz", dat=dat)
    print("wrote", out / "steinmetz_part0.npz")
