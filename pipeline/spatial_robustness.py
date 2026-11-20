"""Spatial robustness and uncertainty (EquiCity 3.0).

    python pipeline/spatial_robustness.py          (~10 minutes)

v2's headline was "spatial clustering of prices has intensified" (town-level
Moran's I rising from ~0.40 to ~0.58). This script asks whether that finding
survives reasonable changes to how "spatial" is defined, and how uncertain
each yearly number is.

1. SPECIFICATION GRID (town level, 1995-2026). Moran's I of the location
   premium every year under: k-nearest neighbours k = 3..10, distance bands
   4/6/8 km, inverse-distance weights, each binary and row-standardised.
   Plus a balanced panel (towns present every year): 25 specifications in
   all. For each: Sen's slope per decade, Mann-Kendall trend test,
   and the change between 1995-99 and 2022-26 averages. Yearly values are
   serially correlated, so the trend test is also run with the Hamed-Rao
   variance correction and with trend-free pre-whitening (3.2).

2. A DIFFERENT UNIT: blocks. Block location premia pooled over 3-year
   windows (1996-98 ... 2024-26), Moran's I with 0.5, 1 and 2 km bands.

3. SAMPLING UNCERTAINTY. 100 bootstrap resamples of transactions within
   each year; every resample re-fits the hedonic regression, recomputes the
   town premia, Moran's I and the Gi* hot/cold labels. Gives 95% intervals
   for the premia and for I, and the share of resamples in which each town
   is a significant hot or cold spot (Gi* stability).

4. INEQUALITY DECOMPOSITION over time, by flat type: Theil T of OBSERVED
   town median prices split into between- and within-region parts.
   (Constant-quality prices differ across flat types only by a constant in
   each year, and Theil is scale-invariant, so the decomposition is
   identical for every flat type; observed medians are where flat type can
   show a difference.)

Outputs: processed_data/analytics/robust_*.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dashboard"))
from common import ANALYTICS_DIR, REGION_OF_TOWN, load_transactions  # noqa: E402
from hedonic import fit_year, fixed_town_weights, weighted_town_effect  # noqa: E402
from analysis.spatial import getis_ord_gi, haversine_matrix, knn_weights, theil_decomposition  # noqa: E402
from data.loader import TOWN_COORDS  # noqa: E402

YEARS = list(range(1995, 2027))
N_BOOT = 100
COORDS = os.path.join(HERE, "..", "raw_data", "hdb_block_coordinates.csv")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def weights_grid(coords):
    d = haversine_matrix(coords)
    out = {}
    for k in range(3, 11):
        out[f"kNN k={k}"] = knn_weights(coords, k)
    for band in (4, 6, 8):
        W = ((d <= band) & (d > 0)).astype(float)
        nearest = np.argsort(d + np.eye(len(d)) * 1e9, axis=1)[:, 0]
        for i in np.where(W.sum(1) == 0)[0]:  # isolated town: link to its nearest neighbour
            W[i, nearest[i]] = 1.0
        out[f"Distance band {band} km"] = W
    W = 1 / (d + np.eye(len(d)))
    np.fill_diagonal(W, 0)
    out["Inverse distance"] = W
    res = {}
    for name, W in out.items():
        res[f"{name} | binary" if not name.startswith("Inverse") else f"{name} | raw"] = W
        res[f"{name} | row-standardised"] = W / W.sum(1, keepdims=True)
    return res


def morans_I(x, W):
    z = x - x.mean()
    return len(x) / W.sum() * (z @ W @ z) / (z @ z)


def morans_perm_p(x, W, n=499, seed=0):
    rng = np.random.default_rng(seed)
    I = morans_I(x, W)
    perms = np.array([morans_I(rng.permutation(x), W) for _ in range(n)])
    return (np.sum(perms >= I) + 1) / (n + 1)


def _mk_s(y):
    n = len(y)
    return float(sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n)))


def _mk_var(y):
    """Variance of S with the tie correction."""
    n = len(y)
    _, t = np.unique(y, return_counts=True)
    return (n * (n - 1) * (2 * n + 5) - np.sum(t * (t - 1) * (2 * t + 5))) / 18


def _z_p(s, var):
    from scipy.stats import norm
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def _acf(x, nlags):
    y = x - x.mean()
    n = len(x)
    acov = np.correlate(y, y, "full")[n - 1:] / n
    return acov[: nlags + 1] / acov[0]


def _sen_detrend(y):
    n = len(y)
    slope = sen_slope(np.arange(n), y)
    return y - np.arange(1, n + 1) * slope, slope


def mann_kendall(y):
    """Original Mann-Kendall test. Assumes the yearly values are independent."""
    y = np.asarray(y, float)
    return _z_p(_mk_s(y), _mk_var(y))


def mann_kendall_hamed_rao(y, alpha=0.05, max_lag=None):
    """Mann-Kendall with the Hamed & Rao (1998) variance correction for serial
    correlation: Var(S) is inflated by n/n*, built from every lag whose
    autocorrelation of the ranks of the Sen-detrended series is significant.
    `max_lag` limits the lags considered (Hamed and Rao suggest 3 for short
    series); None uses every lag, as pymannkendall does by default.
    Matches pymannkendall.hamed_rao_modification_test (tests/test_v3.py)."""
    from scipy.stats import norm, rankdata
    y = np.asarray(y, float)
    n = len(y)
    d, _ = _sen_detrend(y)
    last = n - 1 if max_lag is None else min(max_lag, n - 1)
    r = _acf(rankdata(d), last)
    bound = norm.ppf(1 - alpha / 2) / np.sqrt(n)
    sni = sum((n - k) * (n - k - 1) * (n - k - 2) * r[k] for k in range(1, last + 1) if abs(r[k]) > bound)
    ratio = 1 + 2 / (n * (n - 1) * (n - 2)) * sni
    return _z_p(_mk_s(y), _mk_var(y) * ratio)


def mann_kendall_tfpw(y):
    """Trend-free pre-whitening (Yue et al. 2002): remove the Sen trend, take out
    the lag-1 autocorrelation, add the trend back, then run the ordinary test."""
    y = np.asarray(y, float)
    n = len(y)
    d, slope = _sen_detrend(y)
    r1 = _acf(d, 1)[1]
    w = d[1:] - r1 * d[:-1]
    w = w + np.arange(1, n) * slope
    return _z_p(_mk_s(w), _mk_var(w))


def mann_kendall_prewhitened(y):
    """Plain pre-whitening (von Storch 1995) on the raw series. Known to strip
    part of a genuine trend along with the autocorrelation, so it is
    conservative; reported for completeness, not as the headline adjustment."""
    y = np.asarray(y, float)
    r1 = _acf(y, 1)[1]
    w = y[1:] - r1 * y[:-1]
    return _z_p(_mk_s(w), _mk_var(w))


def lag1_autocorrelation_detrended(y):
    d, _ = _sen_detrend(np.asarray(y, float))
    return float(_acf(d, 1)[1])


def trend_summary(g):
    """One row per specification: 1995-99 vs 2022-26 change, Sen's slope, and
    the trend test with and without corrections for serial correlation in
    the yearly Moran's I series (adjacent years are not independent)."""
    out = []
    for spec, s in g.groupby("spec"):
        s = s.sort_values("year")
        y = s["I"].to_numpy(float)
        z, p = mann_kendall(y)
        _, p_hr = mann_kendall_hamed_rao(y)
        _, p_tf = mann_kendall_tfpw(y)
        _, p_pw = mann_kendall_prewhitened(y)
        out.append({"spec": spec, "I_1995_99": s[s.year <= 1999]["I"].mean(),
                    "I_2022_26": s[s.year >= 2022]["I"].mean(),
                    "change": s[s.year >= 2022]["I"].mean() - s[s.year <= 1999]["I"].mean(),
                    "sen_slope_per_decade": sen_slope(s["year"].to_numpy(), y) * 10,
                    "mann_kendall_z": z, "mann_kendall_p": p,
                    "lag1_acf_detrended": lag1_autocorrelation_detrended(y),
                    "mk_hamed_rao_p": p_hr, "mk_tfpw_p": p_tf, "mk_prewhitened_p": p_pw})
    return pd.DataFrame(out)


def sen_slope(x, y):
    sl = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))]
    return float(np.median(sl))


def town_premia(fit, weights, towns):
    nat = weighted_town_effect(fit, weights)
    return np.array([fit.town_effect(t) - nat for t in towns])


def balanced_panel_rows(fits, weights):
    """Same statistic on the towns present (30+ sales) in EVERY year, so new
    towns entering the sample (Punggol, Sengkang, Sembawang...) cannot drive
    the trend mechanically."""
    present = None
    for y in YEARS:
        f = fits[y]
        ok = {t for t in f.towns if t in TOWN_COORDS and f.n_by_town.get(t, 0) >= 30}
        present = ok if present is None else present & ok
    towns = sorted(present)
    coords = np.array([TOWN_COORDS[t] for t in towns])
    W = knn_weights(coords, 5)
    return [{"year": y, "spec": f"kNN k=5 | balanced panel ({len(towns)} towns)",
             "I": morans_I(town_premia(fits[y], weights, towns), W), "n_towns": len(towns)} for y in YEARS]


def main():
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    df = load_transactions()
    weights = fixed_town_weights(df)
    fits = {y: fit_year(df[df["year"] == y]) for y in YEARS}
    log("fitted")

    # ---------------- 1. specification grid --------------------------------
    rows = []
    for y in YEARS:
        f = fits[y]
        towns = [t for t in f.towns if t in TOWN_COORDS and f.n_by_town.get(t, 0) >= 30]
        coords = np.array([TOWN_COORDS[t] for t in towns])
        x = town_premia(f, weights, towns)
        for name, W in weights_grid(coords).items():
            rows.append({"year": y, "spec": name, "I": morans_I(x, W), "n_towns": len(towns)})
    rows += balanced_panel_rows(fits, weights)
    # Summarise the values exactly as written to robust_moran_specs.csv (4 dp), so
    # `--summary-only` and trend_sensitivity.py, which read that file, agree to the digit.
    g = pd.DataFrame(rows).round(4)
    summ = trend_summary(g)
    g.round(4).to_csv(os.path.join(ANALYTICS_DIR, "robust_moran_specs.csv"), index=False)
    summ.round(4).to_csv(os.path.join(ANALYTICS_DIR, "robust_moran_summary.csv"), index=False)
    log(f"spec grid: {len(summ)} specs; rising in {(summ['change'] > 0).sum()}, "
        f"MK p<0.05 & rising in {((summ['mann_kendall_p'] < 0.05) & (summ['mann_kendall_z'] > 0)).sum()}, "
        f"after Hamed-Rao in {((summ['mk_hamed_rao_p'] < 0.05) & (summ['change'] > 0)).sum()}")

    # ---------------- 2. block level ----------------------------------------
    c = pd.read_csv(COORDS)
    c = c[c["coord_ok"]]
    c["block_key"] = c["street_name"].astype(str) + " | " + c["block"].astype(str)
    parts = []
    for y in range(1996, 2027):
        f = fits[y]
        gdf = df[df["year"] == y]
        nat = weighted_town_effect(f, weights)
        r = gdf["log_price"].to_numpy(float) - f.predict_log(gdf)
        tr = gdf["town"].astype(str).map(lambda t: f.town_effect(t) - nat).to_numpy()
        parts.append(pd.DataFrame({"block_key": gdf["block_key"].to_numpy(), "year": y, "prem": r + tr}))
    bp = pd.concat(parts)
    bp["window"] = ((bp["year"] - 1996) // 3) * 3 + 1996
    brow = []
    for w, s in bp.groupby("window"):
        b = s.groupby("block_key")["prem"].agg(["mean", "size"])
        b = b[b["size"] >= 5].join(c.set_index("block_key")[["lat", "lng"]], how="inner")
        xy = np.column_stack([np.radians(b["lat"]) * 6371, np.radians(b["lng"]) * 6371 * np.cos(np.radians(1.35))])
        tree = cKDTree(xy)
        x = b["mean"].to_numpy()
        for band in (0.5, 1.0, 2.0):
            pr = tree.query_pairs(band, output_type="ndarray")
            n = len(b)
            W = sparse.csr_matrix((np.ones(2 * len(pr)), (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])),
                                  shape=(n, n))
            z = x - x.mean()
            I = n / W.sum() * (z @ (W @ z)) / (z @ z)
            end = min(w + 2, int(bp["year"].max()))
            brow.append({"window": f"{w}-{end}" if end > w else f"{w} (partial)", "band_km": band, "I": I,
                         "n_blocks": n})
    pd.DataFrame(brow).round(4).to_csv(os.path.join(ANALYTICS_DIR, "robust_block_moran.csv"), index=False)
    log("block level done")

    # ---------------- 3. bootstrap ------------------------------------------
    rng = np.random.default_rng(42)
    prem_rows, moran_rows, gi_rows = [], [], []
    for y in YEARS:
        dy = df[df["year"] == y].reset_index(drop=True)
        f0 = fits[y]
        towns = [t for t in f0.towns if t in TOWN_COORDS and f0.n_by_town.get(t, 0) >= 30]
        coords = np.array([TOWN_COORDS[t] for t in towns])
        Wm = knn_weights(coords, 5)
        Wg = knn_weights(coords, 5, include_self=True)
        B = np.empty((N_BOOT, len(towns)))
        Is = np.empty(N_BOOT)
        for b in range(N_BOOT):
            fb = fit_year(dy.iloc[rng.integers(0, len(dy), len(dy))])
            B[b] = [fb.town_effect(t) - weighted_town_effect(fb, weights) for t in towns]
            Is[b] = morans_I(B[b], Wm)
            if y == YEARS[-1] or y in (2000, 2010, 2019):
                lab = getis_ord_gi(B[b], Wg, n_perm=499, seed=b)["label"].to_numpy()
                for t, l in zip(towns, lab):
                    gi_rows.append({"year": y, "town": t, "rep": b, "label": l})
        x0 = town_premia(f0, weights, towns)
        for i, t in enumerate(towns):
            prem_rows.append({"year": y, "town": t, "premium_pct": (np.exp(x0[i]) - 1) * 100,
                              "lo95_pct": (np.exp(np.percentile(B[:, i], 2.5)) - 1) * 100,
                              "hi95_pct": (np.exp(np.percentile(B[:, i], 97.5)) - 1) * 100})
        moran_rows.append({"year": y, "I": morans_I(x0, Wm), "lo95": np.percentile(Is, 2.5),
                           "hi95": np.percentile(Is, 97.5), "p_perm": morans_perm_p(x0, Wm)})
        log(f"bootstrap {y}")
    pd.DataFrame(prem_rows).round(3).to_csv(os.path.join(ANALYTICS_DIR, "robust_premium_ci.csv"), index=False)
    pd.DataFrame(moran_rows).round(4).to_csv(os.path.join(ANALYTICS_DIR, "robust_moran_ci.csv"), index=False)
    gi = pd.DataFrame(gi_rows)
    stab = (gi.assign(hot=gi["label"].str.startswith("Hot"), cold=gi["label"].str.startswith("Cold"))
              .groupby(["year", "town"]).agg(share_hot_pct=("hot", lambda v: v.mean() * 100),
                                             share_cold_pct=("cold", lambda v: v.mean() * 100)).reset_index())
    stab.round(1).to_csv(os.path.join(ANALYTICS_DIR, "robust_gi_stability.csv"), index=False)

    # Gi* label agreement across k for the latest year
    y = YEARS[-1]
    f = fits[y]
    towns = [t for t in f.towns if t in TOWN_COORDS and f.n_by_town.get(t, 0) >= 30]
    coords = np.array([TOWN_COORDS[t] for t in towns])
    x0 = town_premia(f, weights, towns)
    krows = []
    for k in range(3, 11):
        lab = getis_ord_gi(x0, knn_weights(coords, k, include_self=True), n_perm=4999)["label"]
        krows += [{"k": k, "town": t, "label": l} for t, l in zip(towns, lab)]
    pd.DataFrame(krows).to_csv(os.path.join(ANALYTICS_DIR, "robust_gi_by_k.csv"), index=False)

    # ---------------- 4. Theil by flat type, observed medians ---------------
    obs = (df[df["year"] >= 1995].groupby(["year", "flat_type", "town"], observed=True)["resale_price"]
             .agg(["median", "size"]).reset_index())
    obs = obs[(obs["size"] >= 20) & obs["flat_type"].isin(["3-Room", "4-Room", "5-Room", "Executive"])]
    trow = []
    for (yy, ft), s in obs.groupby(["year", "flat_type"], observed=True):
        if len(s) < 10:
            continue
        d = theil_decomposition(s["median"].to_numpy(float), s["town"].astype(str).map(REGION_OF_TOWN).to_numpy())
        trow.append({"year": yy, "flat_type": ft, **d, "n_towns": len(s)})
    pd.DataFrame(trow).round(5).to_csv(os.path.join(ANALYTICS_DIR, "robust_theil_by_type.csv"), index=False)
    log("done")


def summary_only():
    """Recompute robust_moran_summary.csv from robust_moran_specs.csv without
    re-fitting anything (the yearly Moran's I values are unchanged)."""
    g = pd.read_csv(os.path.join(ANALYTICS_DIR, "robust_moran_specs.csv"))
    summ = trend_summary(g)
    summ.round(4).to_csv(os.path.join(ANALYTICS_DIR, "robust_moran_summary.csv"), index=False)
    return summ


if __name__ == "__main__":
    if "--summary-only" in sys.argv:
        print(summary_only().round(4).to_string())
        sys.exit(0)
    main()
