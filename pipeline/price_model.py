"""EquiCity 2.0 price model: features, target, training and intervals.

Shared by the backtest (pipeline/audit_price_model.py) and by the dashboard,
so the model that is evaluated is byte-for-byte the model that is served.

THE CORE CHANGE: PREDICT RELATIVE TO THE MARKET, NOT IN RAW DOLLARS
--------------------------------------------------------------------
The v1 random forest predicted raw resale price with `year` as a feature.
Tree ensembles cannot extrapolate: for any year past the end of training,
the forest predicts as if it were still the last training year. HDB prices
rose ~55% between 2019 and 2025, so a model trained to 2018 under-predicted
everything after it. That is why v1's chronological test R^2 was 0.33 with a
$120k mean error, while the (leaky) random split looked like 0.94.

v2 splits the problem in two:
  1. A market level: the national constant-quality price index from the
     hedonic backbone (pipeline/hedonic.py), known up to "today".
  2. A cross-sectional model of how far a given flat sits above or below
     that market level:  target = log(price) - log(index[month]).

At prediction time the latest known index level is used. In the backtest
the index is CARRIED FORWARD from the last month before the forecast origin,
so the evaluation never sees market information from the future.

Features: town, flat type, flat model (categorical, native HGB support),
floor area, storey, remaining lease, lease commencement year, sale time,
plus within-town location signals:
  te_street / te_block  smoothed target encodings (comparable-sales level
                        on the same street / in the same block), computed
                        out-of-fold on training rows so they cannot leak
                        the row's own price.
  lat, lng, dist_mrt_km, dist_cbd_km   if block coordinates are available
                        (raw_data/hdb_block_coordinates.csv, OneMap).
"""

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

CAT_COLS = ["town", "flat_type", "flat_model"]
NUM_COLS = ["floor_area_sqm", "storey_mid", "remaining_lease",
            "lease_commence_date", "t"]
TE_COLS = ["te_street", "te_block"]
GEO_COLS = ["lat", "lng", "dist_mrt_km", "dist_cbd_km"]

TE_M_STREET = 30.0   # pseudo-count shrinking a street toward its town
TE_M_BLOCK = 10.0    # pseudo-count shrinking a block toward its street
CBD = (1.2837, 103.8510)  # Raffles Place

HGB_PARAMS = dict(max_iter=500, learning_rate=0.08, max_leaf_nodes=63,
                  min_samples_leaf=40, l2_regularization=1.0,
                  early_stopping=False, random_state=42)
Q_PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                min_samples_leaf=60, early_stopping=False, random_state=42)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_geo(df: pd.DataFrame, coords: pd.DataFrame | None, stations: pd.DataFrame | None):
    """Attach block coordinates and distances. `coords` indexed by block_key
    with lat/lng; `stations` the dated station table from rail_history.

    dist_mrt_km is TIME-CORRECT (v3): distance to the nearest MRT/LRT station
    that was open in the month of sale. v2 used today's network for every
    sale, crediting e.g. a 2010 sale with a 2015 Downtown Line station."""
    if coords is None:
        return df
    from rail_history import nearest_open_distance
    df = df.join(coords[["lat", "lng"]], on="block_key")
    lat, lng = df["lat"].to_numpy(float), df["lng"].to_numpy(float)
    df["dist_cbd_km"] = haversine_km(lat, lng, CBD[0], CBD[1]).astype("float32")
    if stations is not None and len(stations):
        df["dist_mrt_km"] = nearest_open_distance(lat, lng, df["month_idx"].to_numpy(), stations).astype("float32")
    return df


def index_for(df: pd.DataFrame, log_index: pd.Series, carry_from: int | None = None):
    """log(index) for each row's month. If `carry_from` is given, every month
    after it uses the index value AT carry_from -- no future market
    information. Subtract from log(price) to get the model target."""
    m = df["month_idx"].to_numpy()
    if carry_from is not None:
        m = np.minimum(m, carry_from)
    return log_index.reindex(m).to_numpy()


class TargetEncoder:
    """Hierarchical smoothed mean of v = deflated log price per sqm:
    town -> street -> block. Fit on training rows; `fit_transform_oof`
    gives out-of-fold encodings for the training rows themselves."""

    def __init__(self, m_street=TE_M_STREET, m_block=TE_M_BLOCK):
        self.m_street, self.m_block = m_street, m_block

    @staticmethod
    def _v(df, y_defl):
        return y_defl - np.log(df["floor_area_sqm"].to_numpy(float))

    def _stats(self, df, v):
        d = pd.DataFrame({"town": df["town"].astype(str).to_numpy(),
                          "street": df["street_name"].astype(str).to_numpy(),
                          "block": df["block_key"].astype(str).to_numpy(), "v": v})
        town = d.groupby("town")["v"].mean()
        st = d.groupby(["town", "street"])["v"].agg(["sum", "count"])
        st_town = town.reindex(st.index.get_level_values(0)).to_numpy()
        st["enc"] = (st["sum"] + self.m_street * st_town) / (st["count"] + self.m_street)
        street = st["enc"].droplevel(0)
        street = street[~street.index.duplicated()]
        bl = d.groupby(["street", "block"])["v"].agg(["sum", "count"])
        bl_street = street.reindex(bl.index.get_level_values(0)).to_numpy()
        bl["enc"] = (bl["sum"] + self.m_block * bl_street) / (bl["count"] + self.m_block)
        block = bl["enc"].droplevel(0)
        return town, street, block

    def fit(self, df, y_defl):
        self.town_, self.street_, self.block_ = self._stats(df, self._v(df, y_defl))
        self.global_ = float(self.town_.mean())
        return self

    def transform(self, df):
        town = df["town"].astype(str).map(self.town_).fillna(self.global_).to_numpy()
        st = df["street_name"].astype(str).map(self.street_).to_numpy(float)
        st = np.where(np.isnan(st), town, st)
        bl = df["block_key"].astype(str).map(self.block_).to_numpy(float)
        bl = np.where(np.isnan(bl), st, bl)
        return st.astype("float32"), bl.astype("float32")

    def fit_transform_temporal(self, df, y_defl):
        """Time-ordered encodings for training rows: a sale in month m is
        encoded using ONLY sales from months before m (same hierarchy and
        smoothing). This mirrors deployment, where a flat is valued from
        earlier comparables only. Rows with no earlier sale on their street
        or block fall back up the hierarchy, as at prediction time."""
        v = self._v(df, y_defl)
        d = pd.DataFrame({"town": df["town"].astype(str).to_numpy(),
                          "street": df["street_name"].astype(str).to_numpy(),
                          "block": df["block_key"].astype(str).to_numpy(),
                          "m": df["month_idx"].to_numpy(), "v": v})

        def prior(keys):
            g = d.groupby(keys + ["m"])["v"].agg(["sum", "count"]).reset_index().sort_values(keys + ["m"])
            g[["cs", "cn"]] = g.groupby(keys)[["sum", "count"]].cumsum().to_numpy()
            g["ps"], g["pn"] = g["cs"] - g["sum"], g["cn"] - g["count"]  # strictly earlier months
            return d[keys + ["m"]].merge(g[keys + ["m", "ps", "pn"]], on=keys + ["m"], how="left")

        gm = d["v"].mean()
        t = prior(["town"])
        town = np.where(t["pn"].to_numpy() > 0, t["ps"] / t["pn"].replace(0, np.nan), gm)
        s_ = prior(["town", "street"])
        street = (s_["ps"].to_numpy() + self.m_street * town) / (s_["pn"].to_numpy() + self.m_street)
        b_ = prior(["street", "block"])
        block = (b_["ps"].to_numpy() + self.m_block * street) / (b_["pn"].to_numpy() + self.m_block)
        self.fit(df, y_defl)
        return street.astype("float32"), block.astype("float32")

    def fit_transform_oof(self, df, y_defl, n_folds=5, seed=42):
        v = self._v(df, y_defl)
        fold = np.random.default_rng(seed).integers(0, n_folds, len(df))
        st_out = np.empty(len(df), dtype="float32")
        bl_out = np.empty(len(df), dtype="float32")
        for k in range(n_folds):
            tr, te = fold != k, fold == k
            enc = TargetEncoder(self.m_street, self.m_block)
            enc.town_, enc.street_, enc.block_ = self._stats(df[tr], v[tr])
            enc.global_ = float(enc.town_.mean())
            st_out[te], bl_out[te] = enc.transform(df[te])
        self.fit(df, y_defl)
        return st_out, bl_out


def make_X(df: pd.DataFrame, categories: dict, use_te=True, use_geo=False):
    cols = CAT_COLS + NUM_COLS + (TE_COLS if use_te else []) + (GEO_COLS if use_geo else [])
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=categories[c])
    for c in cols[len(CAT_COLS):]:
        X[c] = df[c].astype("float32")
    return X


def fit_point(X, y):
    return HistGradientBoostingRegressor(categorical_features="from_dtype", **HGB_PARAMS).fit(X, y)


def fit_quantiles(X, y, alpha=0.2):
    lo = HistGradientBoostingRegressor(loss="quantile", quantile=alpha / 2,
                                       categorical_features="from_dtype", **Q_PARAMS).fit(X, y)
    hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - alpha / 2,
                                       categorical_features="from_dtype", **Q_PARAMS).fit(X, y)
    return lo, hi


def cqr_margin(lo_pred, hi_pred, y_cal, alpha=0.2):
    """Conformalized quantile regression (Romano, Patterson & Candes 2019).
    Returns the additive margin q such that [lo - q, hi + q] has
    finite-sample coverage >= 1 - alpha on exchangeable data."""
    scores = np.maximum(lo_pred - y_cal, y_cal - hi_pred)
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "n": int(len(y_true)),
        "mae": round(float(np.abs(err).mean())),
        "rmse": round(float(np.sqrt((err ** 2).mean()))),
        "mape_pct": round(float(ape.mean() * 100), 2),
        "median_ape_pct": round(float(np.median(ape) * 100), 2),
        "within_5pct": round(float((ape <= 0.05).mean() * 100), 1),
        "within_10pct": round(float((ape <= 0.10).mean() * 100), 1),
        "bias_pct": round(float((err / y_true).mean() * 100), 2),
        "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
    }
