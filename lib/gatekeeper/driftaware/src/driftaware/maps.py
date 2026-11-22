"""Region maps: combine session-level p-values per region and control the false discovery rate."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

__all__ = ["fisher", "bh", "region_map"]


def fisher(p) -> float:
    """Fisher's method for independent p-values."""
    p = np.clip(np.asarray(p, float), 1e-300, 1.0)
    if len(p) == 0:
        return np.nan
    return float(chi2.sf(-2 * np.log(p).sum(), 2 * len(p)))


def bh(p, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of discoveries at FDR q."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    out[order[:k]] = True
    return out


def region_map(df, p_col: str, region_col: str = "region", q: float = 0.05, min_sessions: int = 1):
    """Fisher-combine session p-values per region, then BH across regions.

    df: a pandas DataFrame with one row per session x region. Returns a DataFrame indexed by region with
    columns n_sessions, p_fisher, significant.
    """
    import pandas as pd
    g = df.groupby(region_col)[p_col]
    out = pd.DataFrame({"n_sessions": g.size(), "p_fisher": g.apply(fisher)})
    out = out[out.n_sessions >= min_sessions]
    out["significant"] = bh(out.p_fisher.values, q)
    return out.sort_values("p_fisher")
