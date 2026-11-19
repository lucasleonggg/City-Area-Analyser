"""Model Validation -- how good is the price model, honestly measured?"""

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import store
from views._common import method_note, need

LABELS = {
    "naive": "Naive: last 12 months' town x type median",
    "hedonic_ols": "Linear hedonic regression",
    "rf_v1": "v1 random forest (raw price, year feature)",
    "rf_indexed": "v1 forest, market-indexed target",
    "hgb": "Gradient boosting, indexed",
    "hgb_te": "+ street/block comparable-sales level",
    "hgb_te_geo": "+ block coordinates & rail distance (v2)",
    "hgb_te_geo_temporal": "+ time-ordered encodings, time-correct rail (v3, served)",
}


def show():
    st.title("Model Validation")
    st.markdown(
        "Every model is tested the way it would be used: **trained only on the past, then asked "
        "to price the next six months of real sales**, repeated from eight forecast origins "
        "(January 2019 to January 2026). The market level for those six months is frozen at the "
        "last value known before the origin, so no future information leaks in.")
    if not need("audit_backtest_overall.csv", "audit_backtest_by_origin.csv", "audit_subgroups.csv"):
        return
    from analysis.bundle_io import version_mismatches
    b = store.model_bundle()
    if b is not None:
        mism = version_mismatches(b)
        if mism:
            st.error("This environment differs from the one the model was trained in: "
                     + ", ".join(f"{k} trained {v[0]}, running {v[1]}" for k, v in mism.items())
                     + ". Install the pinned versions: `pip install -r dashboard/requirements.txt`.")
        else:
            st.caption("Environment check: library versions match those the served model was trained with "
                       f"({', '.join(f'{k} {v}' for k, v in b.get('versions', {}).items())}).")
    ov = store.table("audit_backtest_overall.csv")
    fc = ov[ov["mode"] == "forecast"].copy()
    fc["label"] = fc["model"].map(LABELS)
    best = fc.sort_values("mape_pct").iloc[0]
    v1 = fc[fc["model"] == "rf_v1"]
    a, b, c, d = st.columns(4)
    served = fc[fc["model"] == "hgb_te_geo_temporal"]
    best = served.iloc[0] if not served.empty else best
    a.metric("Served model: mean abs. % error", f"{best['mape_pct']:.1f}%")
    if not v1.empty:
        b.metric("v1 mean abs. % error", f"{v1['mape_pct'].iloc[0]:.1f}%")
    c.metric("Within 10% of actual", f"{best['within_10pct']:.0f}%")
    d.metric("Out-of-time R²", f"{best['r2']:.3f}")

    st.subheader("The ablation ladder")
    fig = px.bar(fc.sort_values("mape_pct", ascending=False), x="mape_pct", y="label", orientation="h",
                 text="mape_pct", labels={"mape_pct": "Mean absolute % error (lower is better)", "label": ""},
                 height=380)
    st.plotly_chart(fig, width="stretch")
    show_cols = ["label", "n", "mae", "rmse", "mape_pct", "median_ape_pct", "within_5pct",
                 "within_10pct", "bias_pct", "r2"]
    st.dataframe(fc[show_cols].sort_values("mape_pct"), width="stretch", hide_index=True)
    st.caption("Each rung adds one change. MAE and RMSE are in dollars. Bias is the mean signed "
               "% error (negative = under-predicts).")

    bo = store.table("audit_backtest_by_origin.csv")
    bo = bo[bo["mode"] == "forecast"].assign(label=lambda d: d["model"].map(LABELS))
    fig = px.line(bo, x="origin", y="mape_pct", color="label", markers=True,
                  labels={"mape_pct": "Mean absolute % error", "origin": "Forecast origin", "label": ""},
                  title="Stability across market conditions (2019 flat, 2021-23 boom, 2025-26 plateau)")
    st.plotly_chart(fig, width="stretch")

    now = ov[ov["mode"] == "nowcast"]
    if not now.empty:
        st.subheader("Forecasting the market vs pricing a flat")
        cmp = ov[ov["model"].isin(now["model"])].assign(label=lambda d: d["model"].map(LABELS))
        fig = px.bar(cmp, x="label", y="mape_pct", color="mode", barmode="group",
                     labels={"mape_pct": "Mean absolute % error", "label": ""}, height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("'Nowcast' re-levels the same predictions with the market index actually observed in "
                   "the sale month. The small gap shows most remaining error is about the individual "
                   "flat, not about guessing where the market went.")

    if store.exists("audit_legacy_split.csv"):
        st.subheader("Why v1 scored R² = 0.33 on its chronological test")
        leg = store.table("audit_legacy_split.csv")
        st.dataframe(leg, width="stretch", hide_index=True)
        frozen = leg[leg["mode"].str.contains("frozen at Dec")]["r2"]
        nowc = leg[leg["mode"].str.contains("nowcast")]["r2"]
        fr = f"{frozen.iloc[0]:.2f}" if len(frozen) else "?"
        nw = f"{nowc.iloc[0]:.2f}" if len(nowc) else "?"
        st.markdown(
            "v1's honest test trained to 2018 and predicted 2019-2026 with the market frozen at 2018. "
            f"The served model does **no better** under that setup (R² = {fr}), because no cross-sectional "
            "model can forecast a market rise of more than 50% it has never seen. Give it the observed "
            f"market level and the same 2018-trained model reaches R² = {nw} up to seven years out. "
            "The fix was not a better algorithm; it was not asking the model to forecast the market.")

    st.subheader("Time-ordered vs random target encoding")
    pair = fc[fc["model"].isin(["hgb_te_geo", "hgb_te_geo_temporal"])]
    if len(pair) == 2:
        st.dataframe(pair[show_cols], width="stretch", hide_index=True)
        st.caption("Random out-of-fold encoding lets a 2017 training sale be encoded with 2018 comparables "
                   "from the same block. Time-ordered encoding uses only earlier sales, as the deployed model "
                   "must. The served model uses time-ordered encoding even where it scores about the same, "
                   "because its training then matches how it is used.")

    if store.exists("audit_horizons.csv"):
        st.subheader("Longer horizons: how fast does accuracy decay?")
        hz = store.table("audit_horizons.csv")
        c1, c2 = st.columns(2)
        fig = px.bar(hz, x="bucket", y=["mape_pct", "naive_mape_pct"], barmode="group",
                     labels={"value": "Mean absolute % error", "bucket": "Months after the forecast origin",
                             "variable": ""}, title="Error by horizon (14 half-yearly origins)")
        c1.plotly_chart(fig, width="stretch")
        fig = px.bar(hz, x="bucket", y="coverage80_pct", labels={"coverage80_pct": "80% interval coverage (%)",
                                                                  "bucket": "Months after the forecast origin"},
                     title="Interval coverage by horizon")
        fig.add_hline(y=80, line_dash="dash")
        c2.plotly_chart(fig, width="stretch")
        st.dataframe(hz, width="stretch", hide_index=True)
        st.caption("The intervals are calibrated for the next six months. Further out, the frozen market "
                   "level drifts and coverage falls: for those horizons, use a scenario for the market (see "
                   "Price Predictor), not these intervals.")
    if store.exists("audit_halfyear_origins.csv"):
        ho = store.table("audit_halfyear_origins.csv")
        fig = px.line(ho, x="origin", y=["mape_pct", "coverage80_pct"], markers=True,
                      labels={"value": "%", "origin": "Forecast origin", "variable": ""},
                      title="Next-6-month error and coverage at 14 half-yearly origins")
        st.plotly_chart(fig, width="stretch")

    if store.exists("market_forecast_baselines.csv"):
        st.subheader("Can the market level itself be forecast?")
        mf = store.table("market_forecast_baselines.csv")
        fig = px.bar(mf, x="h", y="mae_pct", color="method", barmode="group",
                     labels={"mae_pct": "Mean absolute error of the index forecast (%)", "h": "Horizon (months)"},
                     title="Three transparent forecasts of the national index, 300 monthly origins 2000-2024")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(mf, width="stretch", hide_index=True)
        st.caption("This is why EquiCity values flats at the current market level and treats the future "
                   "market as a scenario: a year ahead, even the best simple forecast is off by about 5% on "
                   "average and by over 11% one time in ten.")

    st.subheader("Where is the model weaker?")
    sg = store.table("audit_subgroups.csv")
    dim = st.selectbox("Break down by", ["town", "flat_type", "region", "lease_band", "price_band", "origin"])
    sub = sg[sg["dimension"] == dim].sort_values("median_ape_pct")
    c1, c2 = st.columns(2)
    fig = px.bar(sub, x="median_ape_pct", y="group", orientation="h", height=max(320, 22 * len(sub)),
                 labels={"median_ape_pct": "Median absolute % error", "group": ""})
    c1.plotly_chart(fig, width="stretch")
    fig = px.bar(sub, x="coverage80_pct", y="group", orientation="h", height=max(320, 22 * len(sub)),
                 labels={"coverage80_pct": "80% interval coverage (%)", "group": ""})
    fig.add_vline(x=80, line_dash="dash", annotation_text="target 80%")
    c2.plotly_chart(fig, width="stretch")
    st.dataframe(sub, width="stretch", hide_index=True)

    method_note(
        "**Training window**: 10 years before each origin (the v1 replica uses all history, as v1 did).\n\n"
        "**Target**: log(price) minus the log national constant-quality index for the sale month; "
        "the forecast re-adds the last index value known before the origin.\n\n"
        "**Intervals**: conformalized quantile regression (Romano, Patterson & Candès, 2019), "
        "calibrated on the six months before each origin with the same frozen-index rule.\n\n"
        "**Target encodings** (v3): time-ordered; each training sale sees only earlier comparables.\n\n"
        "**Distance to rail** (v3): nearest station open in the month of sale.\n\n"
        "**Random forest settings** in the back-test are 30 trees (v1 used 100) to keep the run "
        "affordable; forest error changes very little beyond ~30 trees.")
