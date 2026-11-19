"""Year-by-year hedonic regressions: the analytical backbone of EquiCity 2.0.

WHY THIS EXISTS
---------------
A town's median resale price mixes two things: how expensive the *location*
is, and what kind of flats happen to have sold there that year. Punggol's
median is pushed up because its stock is young (long leases); Ang Mo Kio's is
pulled down because its stock is old. Comparing raw medians across towns or
over time therefore confounds location with flat mix.

A hedonic regression separates them. For each calendar year y we fit, by OLS:

    log(price) = month effect + town effect + flat-type effect
                 + flat-model effect
                 + b1*lease + b2*lease^2 + b3*log(floor area)
                 + b4*storey + b5*storey^2 + error

Coefficients are re-estimated every year, so the market's valuation of
lease, size and floor level is allowed to drift (it does: see the lease
curve output). From each year's fit we price a fixed REFERENCE FLAT in every
town, which gives a constant-quality price that can be compared across towns
and across years:

    reference flat: 70 years remaining lease, storey 8, the flat type's
    standard floor area (REF_AREA), and the flat type's most common model.

Limitations (stated, not hidden):
  * The town effect is a single % premium shared by all flat types in that
    town-year; there is no town x flat-type interaction.
  * A reference flat can be priced in a town where few or no flats of that
    type sold. `n_town` and `n_cell` are written next to every estimate so
    thin cells can be flagged rather than trusted.
  * OLS on log price gives a geometric-mean (median-like) level. No smearing
    correction is applied; the bias is a uniform ~0.5-1% level shift that does
    not affect any comparison across towns or years.
"""

import numpy as np
import pandas as pd

REF_LEASE = 70.0
REF_STOREY = 8.0
# Typical floor areas (sqm), close to the post-2015 medians for each type.
REF_AREA = {
    "2-Room": 45.0, "3-Room": 67.0, "4-Room": 93.0,
    "5-Room": 113.0, "Executive": 145.0,
}
REF_TOWN = "Ang Mo Kio"
REF_TYPE = "4-Room"
MIN_MODEL_OBS = 30


class YearFit:
    """One year's hedonic fit plus the bookkeeping needed to price any
    reference flat from it."""

    def __init__(self, year, coef, cols, sigma2, xtx_inv, months, towns,
                 types, models, n, n_by_town, n_by_town_type, r2, ref_model):
        self.ref_model = ref_model
        self.year, self.coef, self.cols = year, coef, cols
        self.sigma2, self.xtx_inv = sigma2, xtx_inv
        self.months, self.towns, self.types, self.models = months, towns, types, models
        self.n, self.n_by_town, self.n_by_town_type, self.r2 = n, n_by_town, n_by_town_type, r2
        self._ix = {c: i for i, c in enumerate(cols)}

    def b(self, name):
        i = self._ix.get(name)
        return 0.0 if i is None else float(self.coef[i])

    def town_effect(self, town):
        return self.b(f"town={town}")

    def town_se(self, town):
        i = self._ix.get(f"town={town}")
        if i is None:
            return 0.0  # reference town: effect fixed at 0 by construction
        return float(np.sqrt(self.sigma2 * self.xtx_inv[i, i]))

    def month_effect(self, month_idx):
        return self.b(f"month={month_idx}")

    def mean_month_effect(self):
        return float(np.mean([self.month_effect(m) for m in self.months]))

    def attr_part(self, flat_type, model, lease=REF_LEASE, storey=REF_STOREY, area=None):
        if flat_type not in self.types:
            return float("nan")
        area = REF_AREA.get(flat_type, 93.0) if area is None else area
        if model != self.ref_model and f"model={model}" not in self._ix:
            model = "Other"
        return (self.b(f"type={flat_type}") + self.b(f"model={model}")
                + self.b("lease") * lease + self.b("lease2") * lease ** 2 / 100
                + self.b("log_area") * np.log(area)
                + self.b("storey") * storey + self.b("storey2") * storey ** 2 / 100)

    def predict_log(self, rows: pd.DataFrame, month_idx: int | None = None):
        """Linear hedonic prediction of log price for arbitrary rows. If
        `month_idx` is given every row is priced at that month's effect
        (used to forecast with the last known month)."""
        n = len(rows)
        out = np.full(n, self.b("const"))
        if month_idx is not None:
            out += self.month_effect(month_idx)
        else:
            out += np.array([self.month_effect(m) for m in rows["month_idx"].to_numpy()])
        town = rows["town"].astype(str)
        out += town.map(lambda t: self.b(f"town={t}")).to_numpy()
        out += rows["flat_type"].astype(str).map(lambda t: self.b(f"type={t}")).to_numpy()
        model = rows["flat_model"].astype(str)
        out += model.map(lambda m: self.b(f"model={m}") if (m == self.ref_model or f"model={m}" in self._ix)
                         else self.b("model=Other")).to_numpy()
        L = rows["remaining_lease"].to_numpy(float)
        S = rows["storey_mid"].to_numpy(float)
        out += (self.b("lease") * L + self.b("lease2") * L ** 2 / 100
                + self.b("log_area") * np.log(rows["floor_area_sqm"].to_numpy(float))
                + self.b("storey") * S + self.b("storey2") * S ** 2 / 100)
        return out

    def lease_relative(self, lease, anchor=90.0):
        """Price of a flat with `lease` years left relative to one with
        `anchor` years left, all else equal."""
        f = lambda L: self.b("lease") * L + self.b("lease2") * L ** 2 / 100  # noqa: E731
        return float(np.exp(f(lease) - f(anchor)))


def _design(dfy, ref_month, models_keep, ref_model):
    cols = ["const"]
    blocks = [np.ones((len(dfy), 1))]

    def dummies(series, prefix, levels, ref):
        lv = [l for l in levels if l != ref]
        s = series.astype(str).to_numpy()
        M = np.zeros((len(dfy), len(lv)))
        for j, l in enumerate(lv):
            M[:, j] = s == l
        cols.extend(f"{prefix}={l}" for l in lv)
        blocks.append(M)

    months = sorted(dfy["month_idx"].unique().tolist())
    dummies(dfy["month_idx"], "month", [str(m) for m in months], str(ref_month))
    towns = sorted(dfy["town"].astype(str).unique().tolist())
    dummies(dfy["town"], "town", towns, REF_TOWN if REF_TOWN in towns else towns[0])
    types = sorted(dfy["flat_type"].astype(str).unique().tolist())
    dummies(dfy["flat_type"], "type", types, REF_TYPE if REF_TYPE in types else types[0])
    model = dfy["flat_model"].astype(str).where(
        dfy["flat_model"].astype(str).isin(models_keep), "Other")
    dummies(model, "model", sorted(set(model)), ref_model)

    L = dfy["remaining_lease"].to_numpy(float)
    S = dfy["storey_mid"].to_numpy(float)
    num = np.column_stack([L, L ** 2 / 100, np.log(dfy["floor_area_sqm"].to_numpy(float)),
                           S, S ** 2 / 100])
    cols.extend(["lease", "lease2", "log_area", "storey", "storey2"])
    blocks.append(num)
    # month labels are strings in the dummy names; normalise to int keys
    cols = [c if not c.startswith("month=") else f"month={int(c[6:])}" for c in cols]
    return np.hstack(blocks), cols, months, towns, types


def fit_year(dfy: pd.DataFrame) -> YearFit:
    year = int(dfy["year"].iloc[0])
    vc = dfy["flat_model"].astype(str).value_counts()
    models_keep = set(vc[vc >= MIN_MODEL_OBS].index)
    ref_model = "Model A" if "Model A" in models_keep else vc.index[0]
    months = sorted(dfy["month_idx"].unique().tolist())
    X, cols, months, towns, types = _design(dfy, months[0], models_keep, ref_model)
    y = dfy["log_price"].to_numpy(float)
    XtX = X.T @ X
    Xty = X.T @ y
    # Tiny ridge purely for numerical safety on rank-deficient years
    # (e.g. a flat model that appears in only one town).
    XtX_r = XtX + 1e-8 * np.eye(len(cols))
    coef = np.linalg.solve(XtX_r, Xty)
    resid = y - X @ coef
    dof = max(len(y) - len(cols), 1)
    sigma2 = float(resid @ resid / dof)
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    xtx_inv = np.linalg.inv(XtX_r)
    n_by_town = dfy.groupby("town", observed=True).size().to_dict()
    n_by_town_type = dfy.groupby(["town", "flat_type"], observed=True).size().to_dict()
    return YearFit(year, coef, cols, sigma2, xtx_inv, months, towns, types,
                   sorted(models_keep | {"Other"}), len(dfy),
                   {str(k): int(v) for k, v in n_by_town.items()},
                   {(str(a), str(b)): int(v) for (a, b), v in n_by_town_type.items()},
                   r2, ref_model)


def fit_all_years(df: pd.DataFrame, years=None) -> dict:
    years = sorted(df["year"].unique()) if years is None else years
    return {int(y): fit_year(df[df["year"] == y]) for y in years}


def modal_models(df: pd.DataFrame) -> dict:
    """Most common flat model per flat type over the whole sample, fixed
    across years so the reference flat does not silently change design."""
    return (df.groupby(["flat_type", "flat_model"], observed=True).size()
              .reset_index(name="n").sort_values("n")
              .groupby("flat_type").tail(1)
              .set_index("flat_type")["flat_model"].astype(str).to_dict())


def fixed_town_weights(df: pd.DataFrame, years=range(2015, 2026)) -> dict:
    """Fixed transaction-share weights used to aggregate town effects into a
    national / regional level. Fixed (not year-varying) so that the national
    index is not moved by shifts in WHERE flats sell."""
    sub = df[df["year"].isin(list(years))]
    w = sub.groupby("town", observed=True).size()
    return (w / w.sum()).to_dict()


def weighted_town_effect(fit: YearFit, weights: dict, towns=None) -> float:
    towns = [t for t in (towns or weights) if t in fit.towns]
    if not towns:
        return 0.0
    ws = np.array([weights.get(t, 0.0) for t in towns])
    if ws.sum() == 0:
        return float(np.mean([fit.town_effect(t) for t in towns]))
    return float(np.dot(ws / ws.sum(), [fit.town_effect(t) for t in towns]))


def national_monthly_log_index(fits: dict, weights: dict, ref_models: dict) -> pd.Series:
    """log price of the national reference 4-room flat, month by month.
    Used (a) as the market-level series on the Market Trends page and
    (b) as the deflator for the ML price model."""
    out = {}
    model = ref_models.get(REF_TYPE, "Model A")
    for y, f in sorted(fits.items()):
        base = f.b("const") + weighted_town_effect(f, weights) + f.attr_part(REF_TYPE, model)
        for m in f.months:
            out[m] = base + f.month_effect(m)
    return pd.Series(out, name="log_index").sort_index()
