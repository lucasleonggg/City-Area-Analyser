"""Stage 1 of EquiCity 2.0: audit the price model, then train the one we serve.

    python pipeline/audit_price_model.py            # full audit + production model
    python pipeline/audit_price_model.py --quick    # 2 origins, smoke test

WHAT IS TESTED
--------------
Rolling-origin backtest. For each forecast origin (1 January of 2019..2026):
  * train on the 10 years before the origin (v1 replica: all history),
  * predict every transaction in the following 6 months,
  * the market index used for those 6 months is the last value known BEFORE
    the origin, carried forward. No future information enters.

Models compared (an ablation ladder -- each rung adds one change):
  naive          median price of the same town x flat type over the last 12
                 months (what a person browsing listings does)
  hedonic_ols    last full year's linear hedonic model (pipeline/hedonic.py)
  rf_v1          the v1 production spec: random forest on raw price, with
                 `year` as a feature, trained on all history
  rf_indexed     same forest, but the target is price relative to the index
  hgb            gradient boosting on the indexed target, + flat model and
                 lease cohort, native categoricals
  hgb_te         + street/block comparable-sales encodings
  hgb_te_geo     + block coordinates, distance to rail and to the CBD
                 (only if raw_data/hdb_block_coordinates.csv exists)

Also reported:
  * "nowcast" accuracy for the HGB models -- the same predictions re-levelled
    with the index actually observed in the test month. This is the accuracy
    a valuer has once the current quarter's index is published, and isolates
    the cross-sectional error from the market-level forecasting error.
  * 80% prediction intervals by conformalized quantile regression (CQR),
    calibrated on the 6 months before each origin UNDER THE SAME
    carried-forward index regime as the test, with empirical coverage by
    town and flat type.
  * the v1 headline test (train <= 2018, test > 2018) re-run for comparison.

Outputs (processed_data/analytics/):
  audit_backtest_by_origin.csv, audit_backtest_overall.csv,
  audit_subgroups.csv, audit_coverage.csv, audit_legacy_split.csv,
  audit_predictions_sample.csv.gz
and the production model: processed_data/price_model_v2.joblib
"""

import argparse
import gc
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANALYTICS_DIR, PROCESSED_DIR, ROOT, load_transactions  # noqa: E402
from hedonic import (fit_all_years, fit_year, fixed_town_weights,  # noqa: E402
                     modal_models, national_monthly_log_index)
from price_model import (CAT_COLS, TargetEncoder, add_geo, cqr_margin,  # noqa: E402
                         fit_point, fit_quantiles, index_for, make_X, metrics)

ALPHA = 0.2  # 80% intervals
COORDS_PATH = os.path.join(ROOT, "raw_data", "hdb_block_coordinates.csv")
MODEL_OUT = os.path.join(PROCESSED_DIR, "price_model_v2.joblib")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_stations():
    """Dated station table (see pipeline/rail_history.py)."""
    from rail_history import load_stations as _ls
    return _ls()


def load_coords():
    if not os.path.exists(COORDS_PATH):
        return None
    c = pd.read_csv(COORDS_PATH)
    c = c[c["coord_ok"]]
    c["block_key"] = c["street_name"].astype(str) + " | " + c["block"].astype(str)
    return c.set_index("block_key")[["lat", "lng"]]


def index_before(df, fits, cut, weights, ref_models):
    """National log index using ONLY transactions in months < `cut`: full-year
    fits for earlier years plus, if `cut` falls mid-year, a refit of that
    year on its months before `cut`."""
    Y = 1990 + cut // 12
    use = {y: f for y, f in fits.items() if y < Y}
    part = df[(df["year"] == Y) & (df["month_idx"] < cut)]
    if len(part) > 500:
        use[Y] = fit_year(part)
    return national_monthly_log_index(use, weights, ref_models)


def rf_v1_matrix(d, towns, types):
    X = pd.DataFrame({
        "floor_area_sqm": d["floor_area_sqm"], "storey_mid": d["storey_mid"],
        "remaining_lease": d["remaining_lease"], "year": d["year"].astype("float32")})
    for t in towns:
        X[f"town_{t}"] = (d["town"].astype(str) == t).astype("float32")
    for t in types:
        X[f"type_{t}"] = (d["flat_type"].astype(str) == t).astype("float32")
    return X.to_numpy("float32")


def prepare(train, other, li_train, li_other, use_geo, te_mode="temporal"):
    """Deflate, target-encode the training rows (time-ordered by default,
    or random out-of-fold), and build X for both frames."""
    y_tr = train["log_price"].to_numpy(float) - li_train
    enc = TargetEncoder()
    st, bl = (enc.fit_transform_temporal(train, y_tr) if te_mode == "temporal"
              else enc.fit_transform_oof(train, y_tr))
    tr = train.assign(te_street=st, te_block=bl)
    st_o, bl_o = enc.transform(other)
    ot = other.assign(te_street=st_o, te_block=bl_o)
    return tr, ot, y_tr, enc


TE_MODE = "temporal"


def run_origin(df, fits, ref_models, cats, origin, horizon, window, use_geo, models):
    Y = 1990 + origin // 12
    train = df[(df["month_idx"] >= origin - window * 12) & (df["month_idx"] < origin)]
    test = df[(df["month_idx"] >= origin) & (df["month_idx"] < origin + horizon)]
    weights = fixed_town_weights(train, years=range(Y - window, Y))
    li = index_before(df, fits, origin, weights, ref_models)
    li_full = national_monthly_log_index(fits, weights, ref_models)  # nowcast only
    li_tr = index_for(train, li)
    li_te_carry = index_for(test, li, carry_from=origin - 1)
    li_te_now = index_for(test, li_full)
    preds = {}

    if "naive" in models:
        recent = train[train["month_idx"] >= origin - 12]
        med = recent.groupby(["town", "flat_type"], observed=True)["resale_price"].median()
        med_ft = recent.groupby("flat_type", observed=True)["resale_price"].median()
        key = list(zip(test["town"], test["flat_type"]))
        p = pd.Series(key).map(med.to_dict()).to_numpy(float)
        fb = test["flat_type"].map(med_ft.to_dict()).to_numpy(float)
        preds["naive"] = np.where(np.isnan(p), fb, p)

    if "hedonic_ols" in models:
        f = fits[Y - 1]
        preds["hedonic_ols"] = np.exp(f.predict_log(test, month_idx=max(f.months)))

    if "rf_v1" in models:
        hist = df[df["month_idx"] < origin]
        towns, types = cats["town"], cats["flat_type"]
        rf = RandomForestRegressor(n_estimators=30, max_depth=14, max_samples=min(200_000, len(hist)),
                                   n_jobs=-1, random_state=42)
        rf.fit(rf_v1_matrix(hist, towns, types), hist["resale_price"].to_numpy("float32"))
        preds["rf_v1"] = rf.predict(rf_v1_matrix(test, towns, types))
        del rf, hist; gc.collect()

    if "rf_indexed" in models:
        towns, types = cats["town"], cats["flat_type"]
        Xtr = rf_v1_matrix(train.assign(year=train["t"]), towns, types)
        rf = RandomForestRegressor(n_estimators=30, max_depth=14, max_samples=min(200_000, len(train)),
                                   n_jobs=-1, random_state=42)
        rf.fit(Xtr, train["log_price"].to_numpy(float) - li_tr)
        preds["rf_indexed"] = np.exp(rf.predict(rf_v1_matrix(test.assign(year=test["t"]), towns, types))
                                     + li_te_carry)
        del rf, Xtr; gc.collect()

    now = {}
    prepared = {}
    for name, use_te, geo, mode in [("hgb", False, False, "oof"), ("hgb_te", True, False, "oof"),
                                    ("hgb_te_geo", True, True, "oof"),
                                    ("hgb_te_geo_temporal", True, True, "temporal")]:
        if name not in models or (geo and not use_geo):
            continue
        if mode not in prepared:
            prepared[mode] = prepare(train, test, li_tr, None, use_geo, te_mode=mode)
        tr, te, y_tr, _ = prepared[mode]
        m = fit_point(make_X(tr, cats, use_te, geo), y_tr)
        rel = m.predict(make_X(te, cats, use_te, geo))
        preds[name] = np.exp(rel + li_te_carry)
        now[name] = np.exp(rel + li_te_now)
        del m; gc.collect()
    tr, te, y_tr, _ = prepared.get(TE_MODE) or prepare(train, test, li_tr, None, use_geo, te_mode=TE_MODE)

    # --- CQR intervals for the richest available HGB variant -------------
    final = ("hgb_te_geo_temporal" if TE_MODE == "temporal" else "hgb_te_geo") if use_geo else "hgb_te"
    geo = use_geo
    cal_cut = origin - 6
    cal_tr = train[train["month_idx"] < cal_cut]
    cal = train[train["month_idx"] >= cal_cut]
    li_q = index_before(df, fits, cal_cut, weights, ref_models)
    ctr, ccal, cy, _ = prepare(cal_tr, cal, index_for(cal_tr, li_q), None, geo, te_mode=TE_MODE)
    qlo, qhi = fit_quantiles(make_X(ctr, cats, True, geo), cy, ALPHA)
    Xc = make_X(ccal, cats, True, geo)
    y_cal = cal["log_price"].to_numpy(float) - index_for(cal, li_q, carry_from=cal_cut - 1)
    margin = cqr_margin(qlo.predict(Xc), qhi.predict(Xc), y_cal, ALPHA)
    del qlo, qhi; gc.collect()
    qlo, qhi = fit_quantiles(make_X(tr, cats, True, geo), y_tr, ALPHA)
    Xt = make_X(te, cats, True, geo)
    lo = np.exp(qlo.predict(Xt) - margin + li_te_carry)
    hi = np.exp(qhi.predict(Xt) + margin + li_te_carry)
    del qlo, qhi; gc.collect()

    out = test[["month", "town", "flat_type", "flat_model", "remaining_lease",
                "floor_area_sqm", "region", "resale_price"]].copy()
    out["origin"] = f"{Y}-01"
    for k, v in preds.items():
        out[f"pred_{k}"] = v
    for k, v in now.items():
        out[f"now_{k}"] = v
    out["lo80"], out["hi80"], out["cqr_margin"] = lo, hi, margin
    out["final_model"] = final
    return out


def summarise(all_pred, models):
    rows = []
    for o, g in all_pred.groupby("origin"):
        for m in models:
            col = f"pred_{m}"
            if col in g:
                rows.append({"origin": o, "model": m, "mode": "forecast", **metrics(g["resale_price"], g[col])})
            col = f"now_{m}"
            if col in g:
                rows.append({"origin": o, "model": m, "mode": "nowcast", **metrics(g["resale_price"], g[col])})
    by_origin = pd.DataFrame(rows)
    overall = []
    for m in models:
        for mode, pre in [("forecast", "pred_"), ("nowcast", "now_")]:
            col = pre + m
            if col in all_pred:
                overall.append({"model": m, "mode": mode, **metrics(all_pred["resale_price"], all_pred[col])})
    return by_origin, pd.DataFrame(overall)


def subgroup_tables(all_pred):
    final = all_pred["final_model"].iloc[0]
    p = all_pred.copy()
    p["ape"] = (p[f"pred_{final}"] - p["resale_price"]).abs() / p["resale_price"]
    p["covered"] = (p["resale_price"] >= p["lo80"]) & (p["resale_price"] <= p["hi80"])
    p["width_pct"] = (p["hi80"] - p["lo80"]) / p[f"pred_{final}"] * 100
    p["lease_band"] = pd.cut(p["remaining_lease"], [0, 50, 60, 70, 80, 90, 100],
                             labels=["<50", "50-60", "60-70", "70-80", "80-90", "90+"])
    p["price_band"] = pd.qcut(p["resale_price"], 5, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4", "Q5 (dearest)"])
    rows = []
    for dim in ["town", "flat_type", "region", "lease_band", "price_band", "origin"]:
        for k, g in p.groupby(dim, observed=True):
            if len(g) < 30:
                continue
            rows.append({
                "dimension": dim, "group": str(k), "n": len(g),
                "mape_pct": round(g["ape"].mean() * 100, 2),
                "median_ape_pct": round(g["ape"].median() * 100, 2),
                "bias_pct": round(((g[f"pred_{final}"] - g["resale_price"]) / g["resale_price"]).mean() * 100, 2),
                "coverage80_pct": round(g["covered"].mean() * 100, 1),
                "median_width_pct": round(g["width_pct"].median(), 1),
            })
    return pd.DataFrame(rows)


def legacy_split(df, fits, ref_models, cats, use_geo):
    """Re-run v1's headline test (train <=2018, test >2018) with the v2
    model. Index carried forward from Dec 2018 for 7.5 years -- a market-level
    forecast no cross-sectional model can make -- plus the nowcast."""
    origin = (2019 - 1990) * 12
    train = df[(df["month_idx"] >= origin - 120) & (df["month_idx"] < origin)]
    test = df[df["month_idx"] >= origin]
    weights = fixed_town_weights(train, years=range(2009, 2019))
    li = national_monthly_log_index({y: f for y, f in fits.items() if y < 2019}, weights, ref_models)
    li_full = national_monthly_log_index(fits, weights, ref_models)
    tr, te, y_tr, _ = prepare(train, test, index_for(train, li), None, use_geo, te_mode=TE_MODE)
    m = fit_point(make_X(tr, cats, True, use_geo), y_tr)
    rel = m.predict(make_X(te, cats, True, use_geo))
    y = test["resale_price"].to_numpy(float)
    return pd.DataFrame([
        {"model": "rf_v1 (published)", "mode": "frozen at 2018", "r2": 0.3339, "mae": 120331,
         "rmse": 157240, "mape_pct": None, "n": 193738},
        {"model": "v3 served model", "mode": "index frozen at Dec 2018",
         **metrics(y, np.exp(rel + index_for(test, li, carry_from=origin - 1)))},
        {"model": "v3 served model", "mode": "index updated (nowcast)",
         **metrics(y, np.exp(rel + index_for(test, li_full)))},
    ])


def horizon_backtest(df, fits, ref_models, cats, use_geo, window=10):
    """Half-yearly origins (Jan/Jul 2019-2025), the served model only, test
    window out to 24 months. Shows how error and interval coverage decay as
    the frozen market level ages, and gives twice as many origins as the
    ablation for stability."""
    rows = []
    origins = [(y - 1990) * 12 + m for y in range(2019, 2026) for m in (0, 6)]
    last = int(df["month_idx"].max())
    for origin in origins:
        Y = 1990 + origin // 12
        train = df[(df["month_idx"] >= origin - window * 12) & (df["month_idx"] < origin)]
        test = df[(df["month_idx"] >= origin) & (df["month_idx"] < min(origin + 24, last + 1))]
        weights = fixed_town_weights(train, years=range(Y - window, Y))
        li = index_before(df, fits, origin, weights, ref_models)
        tr, te, y_tr, _ = prepare(train, test, index_for(train, li), None, use_geo, te_mode=TE_MODE)
        point = fit_point(make_X(tr, cats, True, use_geo), y_tr)
        cal_cut = origin - 6
        li_q = index_before(df, fits, cal_cut, weights, ref_models)
        cal_tr, cal = train[train["month_idx"] < cal_cut], train[train["month_idx"] >= cal_cut]
        ctr, ccal, cy, _ = prepare(cal_tr, cal, index_for(cal_tr, li_q), None, use_geo, te_mode=TE_MODE)
        qlo, qhi = fit_quantiles(make_X(ctr, cats, True, use_geo), cy, ALPHA)
        Xc = make_X(ccal, cats, True, use_geo)
        margin = cqr_margin(qlo.predict(Xc), qhi.predict(Xc),
                            cal["log_price"].to_numpy(float) - index_for(cal, li_q, carry_from=cal_cut - 1), ALPHA)
        qlo, qhi = fit_quantiles(make_X(tr, cats, True, use_geo), y_tr, ALPHA)
        Xt = make_X(te, cats, True, use_geo)
        carry = index_for(test, li, carry_from=origin - 1)
        p = np.exp(point.predict(Xt) + carry)
        lo, hi = np.exp(qlo.predict(Xt) - margin + carry), np.exp(qhi.predict(Xt) + margin + carry)
        recent = train[train["month_idx"] >= origin - 12]
        med = recent.groupby(["town", "flat_type"], observed=True)["resale_price"].median().to_dict()
        naive = pd.Series(list(zip(test["town"], test["flat_type"]))).map(med).to_numpy(float)
        y = test["resale_price"].to_numpy(float)
        h = test["month_idx"].to_numpy() - origin
        rows.append(pd.DataFrame({"origin": f"{Y}-{origin % 12 + 1:02d}", "h": h, "y": y, "pred": p,
                                  "lo": lo, "hi": hi, "naive": naive, "n_tr_street": 0}))
        log(f"  horizons origin {Y}-{origin % 12 + 1:02d}: n={len(test):,}")
        del point, qlo, qhi; gc.collect()
    a = pd.concat(rows, ignore_index=True)
    a["bucket"] = pd.cut(a["h"], [-1, 5, 11, 17, 23], labels=["0-5 months", "6-11 months", "12-17 months", "18-23 months"])
    a["ape"] = np.abs(a["pred"] - a["y"]) / a["y"]
    a["ape_naive"] = np.abs(a["naive"] - a["y"]) / a["y"]
    a["covered"] = (a["y"] >= a["lo"]) & (a["y"] <= a["hi"])
    a["signed"] = (a["pred"] - a["y"]) / a["y"]
    by_h = (a.groupby("bucket", observed=True)
              .agg(n=("y", "size"), mape_pct=("ape", lambda v: v.mean() * 100),
                   median_ape_pct=("ape", lambda v: v.median() * 100),
                   naive_mape_pct=("ape_naive", lambda v: np.nanmean(v) * 100),
                   bias_pct=("signed", lambda v: v.mean() * 100),
                   coverage80_pct=("covered", lambda v: v.mean() * 100)).reset_index())
    by_o = (a[a["h"] < 6].groupby("origin")
              .agg(n=("y", "size"), mape_pct=("ape", lambda v: v.mean() * 100),
                   bias_pct=("signed", lambda v: v.mean() * 100),
                   coverage80_pct=("covered", lambda v: v.mean() * 100)).reset_index())
    by_h.round(2).to_csv(os.path.join(ANALYTICS_DIR, "audit_horizons.csv"), index=False)
    by_o.round(2).to_csv(os.path.join(ANALYTICS_DIR, "audit_halfyear_origins.csv"), index=False)
    print(by_h.round(2).to_string())


def market_forecast_baselines(df, fits, ref_models):
    """How well can the MARKET LEVEL itself be forecast? Monthly origins
    2000-2024; the index is rebuilt from data before each origin; forecasts
    at 6, 12 and 24 months from three transparent baselines."""
    weights = fixed_town_weights(df)
    li_full = national_monthly_log_index(fits, weights, ref_models)
    last = int(li_full.index.max())
    rows = []
    for origin in range((2000 - 1990) * 12, (2025 - 1990) * 12):
        li = index_before(df, fits, origin, weights, ref_models)
        hist = li[li.index < origin]
        x0 = hist.iloc[-1]
        d12 = (hist.iloc[-1] - hist.iloc[-13]) / 12
        d36 = (hist.iloc[-1] - hist.iloc[-37]) / 36
        for h in (6, 12, 24):
            tgt = origin - 1 + h
            if tgt > last:
                continue
            actual = li_full.loc[tgt]
            for name, f in [("Random walk (no change)", x0), ("12-month trend", x0 + d12 * h),
                            ("36-month trend", x0 + d36 * h)]:
                rows.append({"origin_month": origin, "h": h, "method": name,
                             "error_pct": (np.exp(f - actual) - 1) * 100})
    r = pd.DataFrame(rows)
    s = (r.groupby(["method", "h"])
          .agg(mae_pct=("error_pct", lambda v: np.mean(np.abs(v))),
               rmse_pct=("error_pct", lambda v: np.sqrt(np.mean(v ** 2))),
               p90_abs_pct=("error_pct", lambda v: np.percentile(np.abs(v), 90)),
               n=("error_pct", "size")).reset_index())
    s.round(2).to_csv(os.path.join(ANALYTICS_DIR, "market_forecast_baselines.csv"), index=False)
    r["origin"] = r["origin_month"].map(lambda m: f"{1990 + m // 12}-{m % 12 + 1:02d}")
    r.round(3).to_csv(os.path.join(ANALYTICS_DIR, "market_forecast_errors.csv.gz"), index=False)
    print(s.round(2).to_string())


def train_production(df, fits, ref_models, cats, use_geo, coords, stations, window=10):
    last = int(df["month_idx"].max())
    origin = last + 1
    train = df[df["month_idx"] >= origin - window * 12]
    Y_last = 1990 + last // 12
    weights = fixed_town_weights(train, years=range(Y_last - window, Y_last + 1))
    li = national_monthly_log_index(fits, weights, ref_models)
    li_tr = index_for(train, li)
    y_tr = train["log_price"].to_numpy(float) - li_tr
    enc = TargetEncoder()
    st, bl = (enc.fit_transform_temporal(train, y_tr) if TE_MODE == "temporal"
              else enc.fit_transform_oof(train, y_tr))
    tr = train.assign(te_street=st, te_block=bl)
    X = make_X(tr, cats, True, use_geo)
    point = fit_point(X, y_tr)

    cal_cut = origin - 6
    cal_tr, cal = train[train["month_idx"] < cal_cut], train[train["month_idx"] >= cal_cut]
    li_c = index_before(df, fits, cal_cut, weights, ref_models)
    ctr, ccal, cy, _ = prepare(cal_tr, cal, index_for(cal_tr, li_c), None, use_geo, te_mode=TE_MODE)
    qlo, qhi = fit_quantiles(make_X(ctr, cats, True, use_geo), cy, ALPHA)
    Xc = make_X(ccal, cats, True, use_geo)
    margin = cqr_margin(qlo.predict(Xc), qhi.predict(Xc),
                        cal["log_price"].to_numpy(float) - index_for(cal, li_c, carry_from=cal_cut - 1), ALPHA)
    qlo, qhi = fit_quantiles(X, y_tr, ALPHA)

    recent = tr[tr["month_idx"] >= last - 11]
    background = make_X(recent.sample(min(400, len(recent)), random_state=7), cats, True, use_geo)
    # per-block lookup for the predictor UI: street, town and (if any) coords
    blocks = (train.groupby("block_key", observed=True)
                   .agg(town=("town", "first"), street_name=("street_name", "first"),
                        block=("block", "first"), n=("resale_price", "size"),
                        lease_commence_date=("lease_commence_date", "median"))
                   .reset_index())
    blocks["town"] = blocks["town"].astype(str)
    blocks["street_name"] = blocks["street_name"].astype(str)
    if coords is not None:
        blocks = blocks.join(coords, on="block_key")
    bundle = {
        "version": "3.0", "point": point, "q_lo": qlo, "q_hi": qhi, "cqr_margin": margin,
        "alpha": ALPHA, "categories": cats, "use_te": True, "use_geo": use_geo,
        "encoder": {"town": enc.town_, "street": enc.street_, "block": enc.block_,
                    "global": enc.global_},
        "log_index": li, "last_month_idx": last, "background": background,
        "blocks": blocks,
        # stations OPEN at the last data month (prediction is at that month)
        "stations": stations.loc[(stations["open_month_idx"] <= last) & (stations["close_month_idx"] > last),
                                 ["lat", "lng"]].to_numpy(float),
        "te_mode": TE_MODE,
        "train_months": [int(train["month_idx"].min()), last], "n_train": int(len(train)),
    }
    sys.path.insert(0, os.path.join(ROOT, "dashboard"))
    from analysis.bundle_io import to_plain
    joblib.dump(to_plain(bundle), MODEL_OUT, compress=3)
    log(f"saved production model -> {MODEL_OUT} ({os.path.getsize(MODEL_OUT) / 1e6:.1f} MB), "
        f"CQR margin {margin:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-geo", action="store_true")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--models", default="naive,hedonic_ols,rf_v1,rf_indexed,hgb,hgb_te,hgb_te_geo,hgb_te_geo_temporal")
    ap.add_argument("--stages", default="backtest,horizons,market,production",
                    help="backtest: annual-origin ablation; horizons: half-yearly origins out to 24 "
                         "months for the served model; market: index-forecast baselines")
    args = ap.parse_args()
    models = args.models.split(",")
    os.makedirs(ANALYTICS_DIR, exist_ok=True)

    df = load_transactions()
    coords = None if args.no_geo else load_coords()
    stations = load_stations()
    use_geo = coords is not None
    df = add_geo(df, coords, stations)
    if use_geo:
        log(f"geo features on: {df['lat'].notna().mean() * 100:.2f}% of rows have block coordinates")
    cats = {c: sorted(df[c].astype(str).unique().tolist()) for c in CAT_COLS}
    fits = fit_all_years(df)
    ref_models = modal_models(df)
    log(f"loaded {len(df):,} rows; hedonic fits ready")

    stages = set(args.stages.split(","))
    if args.skip_backtest:
        stages.discard("backtest")
    if "market" in stages:
        market_forecast_baselines(df, fits, ref_models)
    if "horizons" in stages and not args.quick:
        horizon_backtest(df, fits, ref_models, cats, use_geo)
    if "backtest" in stages:
        years = [2024, 2026] if args.quick else list(range(2019, 2027))
        parts = []
        for Y in years:
            log(f"origin {Y}-01 ...")
            parts.append(run_origin(df, fits, ref_models, cats, (Y - 1990) * 12,
                                    horizon=6, window=10, use_geo=use_geo, models=models))
            g = parts[-1]
            fm = g["final_model"].iloc[0]
            cov = ((g.resale_price >= g.lo80) & (g.resale_price <= g.hi80)).mean()
            log(f"  n={len(g):,}  " + "  ".join(
                f"{m}:{metrics(g.resale_price, g['pred_' + m])['mape_pct']}%"
                for m in models if f"pred_{m}" in g) + f"  cov80({fm})={cov:.3f}")
        allp = pd.concat(parts, ignore_index=True)
        by_origin, overall = summarise(allp, models)
        by_origin.to_csv(os.path.join(ANALYTICS_DIR, "audit_backtest_by_origin.csv"), index=False)
        overall.to_csv(os.path.join(ANALYTICS_DIR, "audit_backtest_overall.csv"), index=False)
        sg = subgroup_tables(allp)
        sg.to_csv(os.path.join(ANALYTICS_DIR, "audit_subgroups.csv"), index=False)
        allp.sample(min(20000, len(allp)), random_state=1).to_csv(
            os.path.join(ANALYTICS_DIR, "audit_predictions_sample.csv.gz"), index=False)
        print(overall.to_string())
        if not args.quick:
            leg = legacy_split(df, fits, ref_models, cats, use_geo)
            leg.to_csv(os.path.join(ANALYTICS_DIR, "audit_legacy_split.csv"), index=False)
            print(leg.to_string())

    if "production" in stages:
        train_production(df, fits, ref_models, cats, use_geo, coords, stations)


if __name__ == "__main__":
    main()
