"""1 · Price Predictor -- how much might this flat sell for, and how sure are we?"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import explain as E
from analysis import store
from views._common import method_note, money, need

STOREY_RANGES = ["01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12", "13 TO 15", "16 TO 18",
                 "19 TO 21", "22 TO 24", "25 TO 27", "28 TO 30", "31 TO 33", "34 TO 36",
                 "37 TO 39", "40 TO 42", "43 TO 45", "46 TO 48", "49 TO 51"]


@st.cache_resource
def _bundle():
    return E.bundle()


@st.cache_data
def _recent():
    return store.table("recent_sales.csv")


@st.cache_data(show_spinner="Explaining the prediction (exact Shapley over 128 coalitions)...")
def _explain(args: tuple):
    x = E.make_row(*args[:6], street=args[6], block=args[7], b=_bundle())
    return E.group_shapley(x, b=_bundle())


def waterfall(res, title, start_label="Typical recent sale", end_label="This flat"):
    c = res["contributions"].sort_values("dollars")
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(c) + ["total"],
        y=[start_label] + c["driver"].tolist() + [end_label],
        x=[res.get("baseline_price", res.get("price_b"))] + c["dollars"].tolist()
          + [res.get("predicted_price", res.get("price_a"))],
        text=[money(res.get("baseline_price", res.get("price_b")))]
             + [f"{d:+,.0f} ({p:+.1f}%)" for d, p in zip(c["dollars"], c["pct_effect"])]
             + [money(res.get("predicted_price", res.get("price_a")))],
        textposition="outside", connector={"line": {"color": "#bbb"}}))
    start = res.get("baseline_price", res.get("price_b"))
    path = start + c["dollars"].cumsum()
    lo = min(start, path.min(), res.get("predicted_price", res.get("price_a")))
    hi = max(start, path.max(), res.get("predicted_price", res.get("price_a")))
    pad = max((hi - lo) * 0.35, 0.03 * hi)
    fig.update_layout(title=title, height=440, margin=dict(l=10, r=40, t=50, b=10), xaxis_title="$",
                      xaxis_range=[lo - pad, hi + pad])
    return fig


def flat_inputs(prefix, rs, b, defaults=None):
    d = defaults or {}
    towns = sorted(b["encoder"]["town"].index)
    c1, c2, c3 = st.columns(3)
    town = c1.selectbox("Town", towns, index=towns.index(d.get("town", "Tampines")), key=f"{prefix}_town")
    rt = rs[rs["town"] == town]
    streets = ["(any street in town)"] + sorted(rt["street_name"].unique())
    street = c2.selectbox("Street", streets, key=f"{prefix}_street")
    street = None if street.startswith("(") else street
    blocks = ["(any block)"] + (sorted(rt[rt["street_name"] == street]["block"].astype(str).unique())
                                if street else [])
    block = c3.selectbox("Block", blocks, key=f"{prefix}_block")
    block = None if block.startswith("(") else block

    sub = rt
    if street:
        sub = sub[sub["street_name"] == street]
    if block:
        sub = sub[sub["block"].astype(str) == block]
    c1, c2, c3 = st.columns(3)
    types = [t for t in ["2-Room", "3-Room", "4-Room", "5-Room", "Executive"] if t in b["categories"]["flat_type"]]
    have = [t for t in types if t in set(sub["flat_type"])] or types
    ft = c1.selectbox("Flat type", have, index=have.index(d.get("flat_type", "4-Room")) if d.get("flat_type", "4-Room") in have else 0,
                      key=f"{prefix}_ft")
    sft = sub[sub["flat_type"] == ft] if not sub[sub["flat_type"] == ft].empty else rt[rt["flat_type"] == ft]
    models = sft["flat_model"].value_counts().index.tolist() or b["categories"]["flat_model"]
    fm = c2.selectbox("Flat model", models, key=f"{prefix}_fm")
    area_default = float(sft["floor_area_sqm"].median()) if not sft.empty else 90.0
    area = c3.number_input("Floor area (sqm)", 30.0, 250.0, float(round(area_default)), 1.0, key=f"{prefix}_area")
    c1, c2 = st.columns(2)
    storey = c1.selectbox("Storey range", STOREY_RANGES, index=2, key=f"{prefix}_storey")
    lc_default = int(sft["lease_commence_date"].median()) if not sft.empty else 1990
    lc = c2.number_input("Lease commencement year", 1966, 2026, lc_default, 1, key=f"{prefix}_lc")
    a, z = storey.split(" TO ")
    return (town, ft, fm, float(area), (int(a) + int(z)) / 2, int(lc), street, block)


def show():
    st.title("Price Predictor")
    st.markdown(
        "Estimate what a resale flat might sell for **at today's market level**, with an "
        "**80% prediction interval** and a breakdown of what drives the estimate.")
    if not need("recent_sales.csv"):
        return
    try:
        b = _bundle()
    except RuntimeError as e:
        st.error(str(e))
        return
    rs = _recent()
    args = flat_inputs("p", rs, b)
    x = E.make_row(*args[:6], street=args[6], block=args[7], b=b)
    pr = E.predict(x, b)

    st.divider()
    a, c, d = st.columns([1.2, 1, 1])
    a.metric("Estimated price", money(round(pr["price"], -3)))
    c.metric("80% interval: low", money(round(pr["lo"], -3)))
    d.metric("80% interval: high", money(round(pr["hi"], -3)))
    st.caption(
        f"Priced at the market level of {E.month_label(b['last_month_idx'])}, the latest month in the data. "
        "In back-testing, 80% intervals built this way contained the actual sale price for roughly 80% "
        "of transactions (see Model Validation for coverage by town and flat type). "
        f"Remaining lease used: {float(x['remaining_lease'].iloc[0]):.1f} years.")
    if store.exists("audit_subgroups.csv"):
        sg = store.table("audit_subgroups.csv")
        row_t = sg[(sg["dimension"] == "town") & (sg["group"] == args[0])]
        row_f = sg[(sg["dimension"] == "flat_type") & (sg["group"] == args[1])]
        if not row_t.empty and not row_f.empty:
            st.caption(f"Back-test accuracy here: typical error {row_t['median_ape_pct'].iloc[0]:.1f}% in "
                       f"{args[0]}, {row_f['median_ape_pct'].iloc[0]:.1f}% for {args[1]} flats (median absolute % error).")

    st.subheader("Valuation today vs the future")
    tab_now, tab_future, tab_market = st.tabs(
        ["Current-market valuation", "Scenario: future value", "Market-level forecast"])
    with tab_now:
        st.markdown(
            f"The estimate above is a **current-market valuation**: what this flat would likely fetch at "
            f"the market level of {E.month_label(b['last_month_idx'])}. It is the model's strength: in "
            "back-tests it is accurate to about 4% on average. It is not a forecast.")
    with tab_future:
        c1, c2 = st.columns(2)
        years = c1.slider("Years ahead", 0.5, 5.0, 1.0, 0.5)
        g = c2.slider("Assumed market change per year (%)", -10.0, 10.0, 0.0, 0.5) / 100
        fut = pr["price"] * (1 + g) ** years
        band = None
        if store.exists("market_forecast_errors.csv.gz"):
            me = store.table("market_forecast_errors.csv.gz")
            me = me[me["method"].str.startswith("Random walk")]
            hs = sorted(me["h"].unique())
            h = min(hs, key=lambda v: abs(v - years * 12))
            e = me[me["h"] == h]["error_pct"]
            band = (np.percentile(e, 10), np.percentile(e, 90), h)
        a, b2, c = st.columns(3)
        a.metric(f"Scenario value in {years:g} years", money(round(fut, -3)))
        if band:
            # forecast error e = forecast/actual - 1, so actual = forecast / (1 + e)
            a_lo, a_hi = fut / (1 + band[1] / 100), fut / (1 + band[0] / 100)
            b2.metric("If the market misses by a typical margin: low", money(round(min(a_lo, a_hi), -3)))
            c.metric("... high", money(round(max(a_lo, a_hi), -3)))
            st.caption(f"The range is the 10th-90th percentile of how far the national index actually moved "
                       f"from a no-change forecast over {band[2]} months, 2000-2024. It reflects market "
                       "uncertainty only; the flat-specific interval above comes on top, and the flat's lease "
                       "will also be shorter by then.")
        st.warning("A scenario, not a prediction: the value follows directly from the market change you assume.")
    with tab_market:
        st.markdown(
            "EquiCity does **not** forecast the HDB market. It measures how hard that is instead: "
            "see *Model Validation → Can the market level itself be forecast?* A year ahead, even the "
            "best of three simple methods misses the national index by about 5% on average.")
        if store.exists("market_forecast_baselines.csv"):
            st.dataframe(store.table("market_forecast_baselines.csv"), width="stretch", hide_index=True)

    st.subheader("Why this estimate?")
    res = _explain(args)
    st.plotly_chart(waterfall(res, "From a typical recent sale to this flat"), width="stretch")
    st.info(E.CAUSATION_NOTE)

    st.subheader("Recent comparable sales")
    town, ft, *_ , street, block = args
    comp = rs[(rs["town"] == town) & (rs["flat_type"] == ft)]
    level = "town"
    if street and not comp[comp["street_name"] == street].empty:
        comp, level = comp[comp["street_name"] == street], "street"
    if block and not comp[comp["block"].astype(str) == block].empty:
        comp, level = comp[comp["block"].astype(str) == block], "block"
    comp = comp.sort_values("month", ascending=False)
    st.caption(f"{len(comp)} {ft} sales in the same {level} over the last 24 months; median "
               f"{money(comp['resale_price'].median()) if len(comp) else '—'}. "
               "Older sales were at a different market level.")
    st.dataframe(comp.head(30), width="stretch", hide_index=True)

    method_note(
        "**Model.** Gradient-boosted trees predict how far this flat sits above or below the "
        "national constant-quality market level; the latest market level is then applied. "
        "Features: town, street and block comparable-sales level, flat type and model, floor area, "
        "storey, remaining lease and lease year" + (", block coordinates and distance to rail and "
        "the CBD" if b["use_geo"] else "") + ".\n\n"
        "**Interval.** Conformalized quantile regression: quantile models give a first band, which "
        "is then widened by the amount needed to cover 80% of the most recent six months of sales, "
        "using only the market information that would have been available at the time.\n\n"
        "**Not included.** Renovation and condition, view, facing, exact unit, upcoming MRT lines, "
        "minimum occupation period effects, or cash-over-valuation negotiations.")
