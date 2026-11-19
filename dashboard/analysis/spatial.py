"""Spatial statistics (EquiCity 2.0, stage 3).

Generic versions of the Gi* code that v1 ran on one variable only, plus the
global test v1 never ran. With 26 towns, any town-level map will show *some*
pattern by chance; Moran's I asks first whether there is spatial clustering
at all, before Gi* is used to say where.

  morans_i(x, W)       global spatial autocorrelation, permutation p-value
  getis_ord_gi(x, W)   local hot/cold spots, conditional permutation p,
                       Benjamini-Hochberg FDR across units
  knn_weights(coords)  binary k-nearest-neighbour weights (Gi* variant
                       includes self; Moran's variant does not)
  inequality(x, groups)  CV, max/min, Gini, and Theil T with an exact
                       between/within-region decomposition

Every function takes plain arrays so it can be applied to any variable:
location premium, affordability index, price growth, the access composite.
"""

import math

import numpy as np
import pandas as pd


def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    lat = np.radians(coords[:, 0])[:, None]
    lng = np.radians(coords[:, 1])[:, None]
    a = (np.sin((lat - lat.T) / 2) ** 2
         + np.cos(lat) * np.cos(lat.T) * np.sin((lng - lng.T) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_weights(coords: np.ndarray, k: int = 5, include_self: bool = False) -> np.ndarray:
    d = haversine_matrix(coords)
    np.fill_diagonal(d, np.inf)
    n = len(coords)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, np.argsort(d[i])[:k]] = 1.0
    if include_self:
        np.fill_diagonal(W, 1.0)
    return W


def morans_i(x, W, n_perm: int = 999, seed: int = 42) -> dict:
    """Global Moran's I with a two-sided permutation p-value. W need not be
    symmetric; it is used as given (row-standardisation does not change the
    permutation test for binary kNN weights)."""
    x = np.asarray(x, float)
    n = len(x)
    z = x - x.mean()
    S0 = W.sum()

    def stat(v):
        return (n / S0) * (v @ W @ v) / (v @ v)

    I = stat(z)
    rng = np.random.default_rng(seed)
    perms = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    EI = -1.0 / (n - 1)
    p = (np.count_nonzero(np.abs(perms - EI) >= abs(I - EI)) + 1) / (n_perm + 1)
    return {"I": float(I), "expected_I": EI, "p_sim": float(p),
            "z_sim": float((I - perms.mean()) / perms.std(ddof=1))}


def _bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def getis_ord_gi(x, W_self, n_perm: int = 9999, seed: int = 42) -> pd.DataFrame:
    """Gi* z-scores with conditional permutation inference (x_i fixed, the
    other n-1 values shuffled) and BH-FDR. `W_self` must include the focal
    unit on its diagonal."""
    x = np.asarray(x, float)
    n = len(x)
    x_bar, s = x.mean(), x.std(ddof=0)

    def gi(i, v):
        w = W_self[i]
        sw = w.sum()
        den = s * math.sqrt((n * (w ** 2).sum() - sw ** 2) / (n - 1))
        return ((w * v).sum() - x_bar * sw) / den if den > 0 else 0.0

    g = np.array([gi(i, x) for i in range(n)])
    rng = np.random.default_rng(seed)
    p = np.empty(n)
    for i in range(n):
        others = np.delete(x, i)
        idx = np.arange(n) != i
        v = np.empty(n)
        v[i] = x[i]
        cnt = 0
        for _ in range(n_perm):
            v[idx] = rng.permutation(others)
            cnt += abs(gi(i, v)) >= abs(g[i])
        p[i] = (cnt + 1) / (n_perm + 1)
    fdr = _bh(p)
    label = np.where(fdr >= 0.05, "Not significant",
                     np.where(g > 0, "Hot spot (FDR<0.05)", "Cold spot (FDR<0.05)"))
    return pd.DataFrame({"gi_z": g, "p_sim": p, "p_fdr": fdr, "label": label})


def gini(x) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def theil_decomposition(x, groups) -> dict:
    """Theil T index of a positive variable across units, decomposed exactly
    into between-group and within-group parts (equal unit weights)."""
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    mu, n = x.mean(), len(x)
    T = float(np.mean(x / mu * np.log(x / mu)))
    between = within = 0.0
    for g in np.unique(groups):
        xg = x[groups == g]
        sg = len(xg) / n * xg.mean() / mu  # group's share of the total
        between += sg * math.log(xg.mean() / mu)
        within += sg * float(np.mean(xg / xg.mean() * np.log(xg / xg.mean())))
    return {"theil": T, "between": between, "within": within,
            "between_share_pct": between / T * 100 if T > 0 else float("nan")}


def inequality(x, groups=None) -> dict:
    x = np.asarray(x, float)
    out = {"cv_pct": float(x.std(ddof=0) / x.mean() * 100),
           "max_min_ratio": float(x.max() / x.min()),
           "p90_p10_ratio": float(np.percentile(x, 90) / np.percentile(x, 10)),
           "gini": gini(x)}
    if groups is not None:
        out.update(theil_decomposition(x, groups))
    return out
