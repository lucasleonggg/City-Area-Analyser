"""Overview -- the research question and what the evidence currently says."""

import pandas as pd
import streamlit as st

from analysis import affordability as A
from analysis import store
from views._common import national_reference_price, need

RQ = ("To what extent does Singapore's HDB resale market exhibit spatial inequality in "
      "housing affordability, and how has that inequality evolved over time?")


def show():
    st.title("EquiCity 3.0")
    st.subheader("Spatial inequality in HDB resale affordability, 2000-2026")
    st.markdown(f"> **Research question.** {RQ}")
    if not need("hedonic_town_year_type.csv", "hedonic_national_month.csv"):
        return

    meta = store.meta()
    st.caption(f"{meta.get('n_transactions', 0):,} resale transactions, {meta.get('first_month', '')} to "
               f"{meta.get('last_month', '')} (Data.gov.sg). Household incomes: SingStat 2000-"
               f"{A.latest_income_year()}.")

    nat = national_reference_price("4-Room").set_index("year")["price"]
    first, last = 2015, int(nat.index.max())
    e0, e1 = A.evaluate(nat[first], first, 50), A.evaluate(nat[last], last, 50)
    st.markdown("#### Headline numbers (constant-quality 4-room flat, median household, HDB loan)")
    a, b, c = st.columns(3)
    a.metric(f"Mortgage share of income, {last}", f"{e1['msr_pct']:.1f}%",
             delta=f"{e1['msr_pct'] - e0['msr_pct']:+.1f} pp since {first}", delta_color="inverse")
    b.metric("Income percentile needed to meet the 30% cap", f"P{e1['required_percentile']:.0f}",
             delta=f"{e1['required_percentile'] - e0['required_percentile']:+.0f} since {first}",
             delta_color="inverse")
    c.metric("Price ÷ annual household income", f"{e1['price_to_annual_income']:.1f}x",
             delta=f"{e1['price_to_annual_income'] - e0['price_to_annual_income']:+.1f} since {first}",
             delta_color="inverse")

    try:
        from views.spatial_inequality import moran_series
        ms = moran_series("location_premium_pct").set_index("year")
        y0 = 2000 if 2000 in ms.index else ms.index.min()
        a, b, c = st.columns(3)
        a.metric(f"Spatial clustering of prices (Moran's I), {last}", f"{ms.loc[last, 'morans_I']:.2f}",
                 help=f"Permutation p = {ms.loc[last, 'p_sim']:.3f}")
        b.metric("Price gap across towns (CV)", f"{ms.loc[last, 'cv_price_pct']:.1f}%",
                 delta=f"{ms.loc[last, 'cv_price_pct'] - ms.loc[y0, 'cv_price_pct']:+.1f} pp since {y0}",
                 delta_color="inverse")
        c.metric("Share of the gap between regions", f"{ms.loc[last, 'theil_between_region_share_pct']:.0f}%",
                 delta=f"{ms.loc[last, 'theil_between_region_share_pct'] - ms.loc[y0, 'theil_between_region_share_pct']:+.0f} pp since {y0}",
                 delta_color="off")
    except Exception as exc:  # spatial tables missing: show the rest anyway
        st.caption(f"Spatial headline unavailable: {exc}")

    if store.exists("audit_backtest_overall.csv"):
        ov = store.table("audit_backtest_overall.csv")
        fc = ov[ov["mode"] == "forecast"].set_index("model")
        best = "hgb_te_geo_temporal" if "hgb_te_geo_temporal" in fc.index else fc["mape_pct"].idxmin()
        a, b, _ = st.columns(3)
        a.metric("Price model error, out of time", f"{fc.loc[best, 'mape_pct']:.1f}%",
                 help="Mean absolute % error pricing the next 6 months of sales, 8 origins 2019-2026.")
        if "rf_v1" in fc.index:
            b.metric("v1 model, same test", f"{fc.loc['rf_v1', 'mape_pct']:.1f}%")

    st.markdown("#### How the platform answers it")
    t4 = store.table("hedonic_town_year_type.csv")
    t4 = t4[(t4["flat_type"] == "4-Room") & (t4["n_town"] >= 30)].rename(columns={"price_ref": "price"})
    w30 = A.affordability_frame(t4, 30).groupby("year")["within_cap"].agg(["sum", "size"])
    st.markdown("#### What the evidence says so far")
    st.markdown(
        "- **Affordability has worsened since 2019, most for lower incomes.** A 30th-percentile household "
        f"can finance the reference 4-room in {int(w30.loc[last, 'sum'])} of {int(w30.loc[last, 'size'])} towns "
        f"in {last}, down from {int(w30.loc[2019, 'sum'])} in 2019.\n"
        "- **Prices have become more clustered between towns** (rising in all 25 spatial specifications "
        "tested, beyond bootstrap noise, and on a balanced panel of towns), and the divide is increasingly "
        "**regional**. Within neighbourhoods (block level) clustering has **not** risen.\n"
        "- **The gap narrowed in the 2020-26 boom** as cheap northern towns rose fastest: clustered, but "
        "less extreme.\n"
        "- **Individual hot spots are fragile**: no town is a significant Gi* hot spot in more than about "
        "half of bootstrap re-fits. See Research & Data for the paper.")
    st.markdown("#### How the platform answers it")
    st.table(pd.DataFrame([
        ("Price Predictor", "What would this flat fetch today, how sure are we, and what if the market moves?"),
        ("Affordability Index", "Can a household at a given income percentile finance a flat here?"),
        ("Scenario Engine", "With my income, savings, CPF and grants, what can I afford, and where?"),
        ("Policy Lab", "What do different rates, LTV limits, grants or MSR caps do, mechanically?"),
        ("Spatial Inequality", "Where are disparities concentrated, are they widening, and how robust is that?"),
        ("Transport Access", "How did rail access change, and what happened to nearby prices?"),
        ("Market Trends", "How have constant-quality prices and the lease discount changed?"),
        ("Explainability", "What drives the model, and why do two flats differ?"),
        ("Model Validation", "How accurate is the model on the future, and where does it break down?"),
        ("Research & Data", "The paper, methods, data dictionary, limitations and downloads."),
    ], columns=["Module", "Question it answers"]).set_index("Module"))

    st.markdown("#### Read before citing")
    st.markdown(
        "- **Constant quality.** Prices are for the same reference flat everywhere (hedonic "
        "regression), so towns are compared on location, not on what kind of flats happened to sell.\n"
        "- **Buyer-side income.** Affordability uses the national household income distribution. "
        "Cross-town differences therefore come from prices; the Census 2020 lens adds local incomes.\n"
        "- **Financing held fixed.** Today's HDB-loan terms are applied to every year unless changed, "
        "to separate the price-income race from policy changes.\n"
        "- **Association, not causation.** Model explanations describe patterns in past sales.\n"
        "- **Not modelled.** CPF grants, savings and sale proceeds, so 'cannot finance' means "
        "'cannot borrow the full loan on income alone', not 'cannot buy'.")
