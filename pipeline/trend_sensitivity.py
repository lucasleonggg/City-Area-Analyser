"""How much should we trust the clustering trend with only 32 yearly values? (3.2)

    python pipeline/trend_sensitivity.py        (~1 minute; needs robust_moran_specs.csv)

The yearly town-level Moran's I series are serially correlated, and 32
observations is a short series for estimating that dependence. The
Hamed-Rao correction (the primary adjusted test, see spatial_robustness.py)
estimates it from the data itself, so this script checks it two ways.

1. LAG SELECTION. Hamed-Rao corrected p-values using every significant lag
   (pymannkendall's default, used in the headline) and only lags 1-3 (Hamed
   and Rao's own suggestion for short series).

2. PERSISTENCE-ONLY NULL, by simulation. For each specification, fit an
   AR(1) to the Sen-detrended series, then simulate 4,000 series of the same
   length with that persistence and NO trend. The simulated p-value is the
   share whose Mann-Kendall S is at least the observed S: "how often would
   persistence alone produce a rise this consistent?" Because AR(1)
   coefficients estimated from 32 points are biased towards zero, it is
   repeated with a deliberately pessimistic persistence (phi = 0.9).

3. RICHER PERSISTENCE. The same simulation (a) with an AR(2) fitted to
   the detrended series, in case dependence is not first-order; (b) with
   the AR(1) coefficient bias-corrected for the short sample and DRAWN from
   its sampling distribution for every simulated series, so uncertainty in
   the persistence estimate itself is carried into the p-value.

4. STRUCTURAL BREAK. Could a one-off level shift, rather than a steady
   rise, explain the series? For each specification: the best single
   mean-shift break (each segment >= 6 years), BIC of a step model against a
   linear trend, and the Sen slope within each segment. A steady rise shows
   positive slopes on both sides of the break; a pure step shows flat ones.

5. SIZE OF THE TESTS. On trendless AR(1) series of length 32, how often does
   each test wrongly report a trend at the 5% level? Shows how far the
   ordinary test over-rejects and whether the corrections fix it.

Outputs: processed_data/analytics/robust_trend_sensitivity.csv,
         processed_data/analytics/robust_trend_breaks.csv,
         processed_data/analytics/robust_trend_test_size.csv
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ANALYTICS_DIR  # noqa: E402
from spatial_robustness import (_acf, _sen_detrend, mann_kendall,  # noqa: E402
                                mann_kendall_hamed_rao, mann_kendall_tfpw, sen_slope)

N_SIM = 4000
SEED = 2026
BURN = 200
MIN_SEGMENT = 6


def mk_s_many(Y):
    """Mann-Kendall S for every row of Y at once."""
    n = Y.shape[1]
    i, j = np.triu_indices(n, 1)
    return np.sign(Y[:, j] - Y[:, i]).sum(axis=1)


def ar1(n_series, n, phi, rng):
    """Trendless AR(1) series. `phi` may be a scalar or one value per series."""
    phi = np.broadcast_to(np.asarray(phi, float), (n_series,))
    e = rng.normal(size=(n_series, n))
    y = np.empty_like(e)
    y[:, 0] = e[:, 0] / np.sqrt(np.maximum(1 - phi ** 2, 1e-6))  # stationary start
    for t in range(1, n):
        y[:, t] = phi * y[:, t - 1] + e[:, t]
    return y


def ar2(n_series, n, phi1, phi2, rng):
    """Trendless AR(2) series, started from zero with a burn-in."""
    e = rng.normal(size=(n_series, n + BURN))
    y = np.zeros_like(e)
    for t in range(2, n + BURN):
        y[:, t] = phi1 * y[:, t - 1] + phi2 * y[:, t - 2] + e[:, t]
    return y[:, BURN:]


def fit_ar1(d):
    """Lag-1 coefficient of a (detrended) series, kept inside (-0.95, 0.95)."""
    return float(np.clip(_acf(np.asarray(d, float), 1)[1], -0.95, 0.95))


def fit_ar2(d):
    """Yule-Walker AR(2), pulled inside the stationarity triangle if needed."""
    r1, r2 = _acf(np.asarray(d, float), 2)[1:3]
    den = 1 - r1 ** 2
    phi1 = r1 * (1 - r2) / den
    phi2 = (r2 - r1 ** 2) / den
    phi2 = float(np.clip(phi2, -0.95, 0.95))
    # stationarity: |phi2| < 1, phi1 + phi2 < 1, phi2 - phi1 < 1
    phi1 = float(np.clip(phi1, phi2 - 0.99, 0.99 - phi2))
    return phi1, phi2


def bias_corrected_ar1(phi, n):
    """Kendall (1954) small-sample correction: E[phi_hat] ~ phi - (1 + 3 phi) / n."""
    return float(np.clip(phi + (1 + 3 * phi) / n, -0.95, 0.95))


def _p_from_sims(y, sims):
    s_obs = mk_s_many(np.asarray(y, float)[None, :])[0]
    return float((np.sum(mk_s_many(sims) >= s_obs) + 1) / (len(sims) + 1))


def ar2_null_p(y, phi1, phi2, rng):
    return _p_from_sims(y, ar2(N_SIM, len(y), phi1, phi2, rng))


def uncertain_ar1_null_p(y, phi_hat, rng):
    """Persistence drawn per simulated series from N(phi_bc, (1 - phi_bc^2) / n)."""
    n = len(y)
    phi_bc = bias_corrected_ar1(phi_hat, n)
    draws = np.clip(rng.normal(phi_bc, np.sqrt((1 - phi_bc ** 2) / n), N_SIM), -0.95, 0.95)
    return _p_from_sims(y, ar1(N_SIM, n, draws, rng)), phi_bc


def structural_break(years, y):
    """Best single mean shift vs a linear trend, and the trend on each side."""
    years = np.asarray(years)
    y = np.asarray(y, float)
    n = len(y)

    def bic(rss, k):
        return n * np.log(rss / n) + k * np.log(n)

    best = None
    for b in range(MIN_SEGMENT, n - MIN_SEGMENT + 1):
        rss = ((y[:b] - y[:b].mean()) ** 2).sum() + ((y[b:] - y[b:].mean()) ** 2).sum()
        if best is None or rss < best[1]:
            best = (b, rss)
    b, rss_step = best
    coef = np.polyfit(years, y, 1)
    rss_lin = ((y - np.polyval(coef, years)) ** 2).sum()
    out = {
        "break_first_year_after": int(years[b]),
        "bic_step": bic(rss_step, 3),     # two means + break date
        "bic_linear_trend": bic(rss_lin, 2),
        "sen_slope_per_decade_before": sen_slope(years[:b], y[:b]) * 10,
        "sen_slope_per_decade_after": sen_slope(years[b:], y[b:]) * 10,
        "mk_p_before": mann_kendall(y[:b])[1],
        "mk_p_after": mann_kendall(y[b:])[1],
    }
    out["preferred_by_bic"] = "linear trend" if out["bic_linear_trend"] < out["bic_step"] else "step"
    return out


def persistence_null_p(y, phi, rng):
    """One-sided Monte Carlo p-value, (k + 1) / (N_SIM + 1), where k is the number
    of trendless AR(1) series with S >= observed S (Davison & Hinkley 1997).
    Its smallest possible value is 1 / (N_SIM + 1) = 0.00025."""
    y = np.asarray(y, float)
    s_obs = mk_s_many(y[None, :])[0]
    sims = mk_s_many(ar1(N_SIM, len(y), phi, rng))
    return float((np.sum(sims >= s_obs) + 1) / (N_SIM + 1))


def sensitivity(g, rng):
    rows = []
    for spec, s in g.groupby("spec"):
        y = s.sort_values("year")["I"].to_numpy(float)
        d, _ = _sen_detrend(y)
        phi = fit_ar1(d)
        phi1, phi2 = fit_ar2(d)
        p_unc, phi_bc = uncertain_ar1_null_p(y, phi, rng)
        rows.append({
            "spec": spec,
            "phi_ar1_detrended": phi,
            "phi_ar1_bias_corrected": phi_bc,
            "ar2_phi1": phi1,
            "ar2_phi2": phi2,
            "mk_original_p": mann_kendall(y)[1],
            "mk_hamed_rao_p_all_lags": mann_kendall_hamed_rao(y)[1],
            "mk_hamed_rao_p_lags_1_3": mann_kendall_hamed_rao(y, max_lag=3)[1],
            "mk_tfpw_p": mann_kendall_tfpw(y)[1],
            "sim_p_fitted_phi": persistence_null_p(y, phi, rng),
            "sim_p_phi_0_9": persistence_null_p(y, 0.9, rng),
            "sim_p_ar2": ar2_null_p(y, phi1, phi2, rng),
            "sim_p_phi_uncertain": p_unc,
        })
    return pd.DataFrame(rows)


def breaks(g):
    rows = []
    for spec, s in g.groupby("spec"):
        s = s.sort_values("year")
        rows.append({"spec": spec, **structural_break(s["year"].to_numpy(), s["I"].to_numpy(float))})
    return pd.DataFrame(rows)


def test_size(rng, n=32, phis=(0.0, 0.3, 0.5, 0.7, 0.8, 0.9), reps=1000):
    """Rejection rate at 5% on trendless AR(1) series (the ideal is 0.05)."""
    rows = []
    for phi in phis:
        Y = ar1(reps, n, phi, rng)
        rej = {"original": 0, "hamed_rao_all_lags": 0, "hamed_rao_lags_1_3": 0, "tfpw": 0}
        for y in Y:
            # Hamed-Rao's variance ratio can go negative on strongly negatively
            # autocorrelated draws; like pymannkendall, that p is NaN (not a rejection).
            with np.errstate(invalid="ignore"):
                hr_all = mann_kendall_hamed_rao(y)[1]
                hr_3 = mann_kendall_hamed_rao(y, max_lag=3)[1]
            rej["original"] += mann_kendall(y)[1] < 0.05
            rej["hamed_rao_all_lags"] += hr_all < 0.05
            rej["hamed_rao_lags_1_3"] += hr_3 < 0.05
            rej["tfpw"] += mann_kendall_tfpw(y)[1] < 0.05
        rows.append({"phi": phi, "n_years": n, "reps": reps,
                     **{f"false_trend_rate_{k}": v / reps for k, v in rej.items()}})
    return pd.DataFrame(rows)


def main():
    g = pd.read_csv(os.path.join(ANALYTICS_DIR, "robust_moran_specs.csv"))
    rng = np.random.default_rng(SEED)
    sens = sensitivity(g, rng)
    brk = breaks(g)
    size = test_size(rng)
    # 6 dp: the smallest simulated p-value is 1/(N_SIM+1) = 0.00025, which 4 dp
    # would round to 0.0002, below what the simulation can resolve.
    sens.round(6).to_csv(os.path.join(ANALYTICS_DIR, "robust_trend_sensitivity.csv"), index=False)
    brk.round(4).to_csv(os.path.join(ANALYTICS_DIR, "robust_trend_breaks.csv"), index=False)
    size.round(3).to_csv(os.path.join(ANALYTICS_DIR, "robust_trend_test_size.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(sens.round(4).to_string())
    print(brk.round(4).to_string())
    print(size.to_string())
    for col in ("mk_hamed_rao_p_all_lags", "mk_hamed_rao_p_lags_1_3", "sim_p_fitted_phi", "sim_p_phi_0_9",
                "sim_p_ar2", "sim_p_phi_uncertain"):
        print(f"{col}: {(sens[col] < 0.05).sum()} of {len(sens)} below 0.05 (max {sens[col].max():.4f})")
    both = ((brk["sen_slope_per_decade_before"] > 0) & (brk["sen_slope_per_decade_after"] > 0)).sum()
    print(f"break: linear trend preferred by BIC in {(brk['preferred_by_bic'] == 'linear trend').sum()} of {len(brk)}; "
          f"rising on both sides of the break in {both} of {len(brk)}")


if __name__ == "__main__":
    main()
