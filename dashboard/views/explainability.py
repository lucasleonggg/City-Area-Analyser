"""5 · Explainability -- what drives the model, and why do two flats differ?"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import explain as E
from analysis import store
from views._common import method_note, money, need
from views.predictor import _bundle, _recent, flat_inputs, waterfall

PDP_LABELS = {
    "remaining_lease": "Remaining lease (years)",
    "floor_area_sqm": "Floor area (sqm)",
    "storey_mid": "Storey",
    "dist_mrt_km": "Distance to nearest MRT/LRT station (km)",
    "dist_cbd_km": "Distance to Raffles Place (km)",
}


def show():
    st.title("Explainability")
    st.markdown(
        "What the price model has learned, explained against **the same model that makes the "
        "predictions**, with exact Shapley values over seven drivers.")
    st.warning(
        "**Correlation is not causation.** Every number on this page describes how the model "
        "uses its inputs, learned from past sales. A large 'location' contribution means flats "
        "there have sold for more, holding the model's other inputs fixed. It does not say that "
        "location *causes* the premium (it may stand in for schools, amenities, flat condition or "
        "who chooses to live there), or that changing one feature of a real flat would change its "
        "price by that amount.")
    if not need("global_shapley.csv", "shapley_sample.csv"):
        return

    st.header("Which drivers matter most?")
    g = store.table("global_shapley.csv")
    sh = store.table("shapley_sample.csv")
    c1, c2 = st.columns([1, 1.3])
    fig = px.bar(g.sort_values("mean_abs_dollars"), x="mean_abs_dollars", y="driver", orientation="h",
                 labels={"mean_abs_dollars": "Mean |contribution| ($)", "driver": ""},
                 title="Average size of each driver's effect", height=420)
    c1.plotly_chart(fig, width="stretch")
    fig = px.strip(sh, x="pct_effect", y="driver", color="driver", height=420,
                   labels={"pct_effect": "Contribution to this flat's price (%)", "driver": ""},
                   title="Spread across 300 recent sales (each dot is one flat)",
                   category_orders={"driver": g["driver"].tolist()})
    fig.update_layout(showlegend=False)
    c2.plotly_chart(fig, width="stretch")
    st.caption("Contributions are relative to a typical recent sale. A driver can matter a lot for "
               "individual flats in both directions yet average out near zero.")

    st.header("How does the price respond to one feature?")
    avail = [f for f in PDP_LABELS if store.exists(f"pdp_{f}.csv")]
    cols = st.columns(2)
    for i, f in enumerate(avail):
        p = store.table(f"pdp_{f}.csv")
        fig = px.line(p, x=f, y="price", markers=True, labels={f: PDP_LABELS[f], "price": "Average predicted price ($)"},
                      height=330)
        cols[i % 2].plotly_chart(fig, width="stretch")
    st.caption(
        "Partial dependence: 400 recent sales re-priced with only this feature changed, then "
        "averaged. Where features are correlated (lease with estate age, distance to the CBD with "
        "town) the curve also reflects combinations that rarely occur in reality.")

    if store.exists("model_town_premium.csv"):
        st.header("Two methods, one answer? Location premium")
        mp = store.table("model_town_premium.csv")
        ht = store.table("hedonic_town_year_type.csv")
        ht = ht[(ht["flat_type"] == "4-Room") & (ht["year"] == ht["year"].max())][["town", "location_premium_pct"]]
        both = mp.merge(ht, on="town")
        rho = both["premium_vs_median_pct"].rank().corr(both["location_premium_pct"].rank())
        fig = px.scatter(both, x="location_premium_pct", y="premium_vs_median_pct", text="town",
                         labels={"location_premium_pct": "Hedonic regression premium (%)",
                                 "premium_vs_median_pct": "Gradient-boosting model premium (%)"},
                         height=520, title=f"Same reference flat priced in every town (Spearman rho = {rho:.2f})")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, width="stretch")
        st.caption("A linear regression and a non-linear tree model, built differently, should "
                   "broadly agree on which towns carry a premium. Where they disagree, treat both "
                   "numbers with care.")

    st.header("Why might two similar flats have different predicted prices?")
    st.markdown("Set up two flats. The contributions below add up exactly to the difference "
                "between their predicted prices (flat B is the baseline).")
    try:
        b = _bundle()
    except RuntimeError as e:
        st.error(str(e))
        return
    rs = _recent()
    st.subheader("Flat A")
    a_args = flat_inputs("a", rs, b, {"town": "Queenstown"})
    st.subheader("Flat B")
    b_args = flat_inputs("b", rs, b, {"town": "Clementi"})
    xa = E.make_row(*a_args[:6], street=a_args[6], block=a_args[7], b=b)
    xb = E.make_row(*b_args[:6], street=b_args[6], block=b_args[7], b=b)
    res = E.compare(xa, xb, b)
    k1, k2, k3 = st.columns(3)
    k1.metric("Flat A", money(round(res["price_a"], -3)))
    k2.metric("Flat B", money(round(res["price_b"], -3)))
    k3.metric("Difference", money(round(res["price_a"] - res["price_b"], -3)))
    res_plot = {"price_b": res["price_b"], "price_a": res["price_a"], "contributions": res["contributions"]}
    st.plotly_chart(waterfall(res_plot, "From flat B to flat A", "Flat B", "Flat A"), width="stretch")

    method_note(E.__doc__.split("WHAT THESE NUMBERS ARE NOT")[0])
