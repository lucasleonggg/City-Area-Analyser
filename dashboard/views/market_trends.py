"""4 · Market Trends -- how have prices changed, where, and for which flats?"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import store
from analysis import temporal as T
from data.loader import get_flat_types, get_town_list, load_hdb_data
from views._common import FLAT_TYPES, method_note, need


def show():
    st.title("Market Trends")
    st.markdown(
        "Prices here are **constant-quality**: each year, a hedonic regression prices the same "
        "reference flat (70 years lease left, storey 8, the flat type's standard size and model) "
        "in every town. Raw medians move when the *mix* of flats sold changes; these do not.")
    if not need("hedonic_national_month.csv", "hedonic_town_year_type.csv", "lease_curve.csv"):
        return

    tabs = st.tabs(["National index", "Towns & regions", "Unusual growth",
                    "Old vs new flats (lease)", "Raw medians"])

    # ---- national ----
    with tabs[0]:
        v = T.national_quarterly_vs_official()
        fig = go.Figure()
        fig.add_scatter(x=v["quarter"], y=v["equicity"], name="EquiCity constant-quality index")
        if v["official_rpi"].notna().any():
            fig.add_scatter(x=v["quarter"], y=v["official_rpi"], name="HDB official Resale Price Index")
        fig.update_layout(title="National resale price index (1Q2009 = 100)", height=460,
                          yaxis_title="Index", xaxis=dict(nticks=20))
        st.plotly_chart(fig, width="stretch")
        both = v.dropna(subset=["official_rpi"])
        if len(both) > 8:
            d_ours = np.diff(np.log(both["equicity"].to_numpy()))
            d_off = np.diff(np.log(both["official_rpi"].astype(float).to_numpy()))
            corr = np.corrcoef(d_ours, d_off)[0, 1]
            last = both.iloc[-1]
            a, b, c = st.columns(3)
            a.metric("Correlation of quarterly changes", f"{corr:.3f}")
            b.metric(f"EquiCity, {last['quarter']}", f"{last['equicity']:.1f}")
            c.metric(f"Official RPI, {last['quarter']}", f"{float(last['official_rpi']):.1f}")
            st.caption(
                "This is a validation, not a replacement: HDB's index uses stratified hedonic "
                "regression on its own registration data. Built independently from the public "
                "transaction file, EquiCity's index tracks its turning points closely, which is "
                "evidence the constant-quality method behind every other chart here is sound. "
                f"The two levels drift apart slowly (EquiCity is {(last['equicity'] / float(last['official_rpi']) - 1) * 100:+.1f}% "
                "in the latest complete quarter), so compare changes rather than levels.")

    # ---- towns ----
    with tabs[1]:
        c1, c2 = st.columns([1, 2])
        ft = c1.selectbox("Flat type", FLAT_TYPES, index=2, key="mt_ft")
        towns = c2.multiselect("Towns", sorted(T.town_prices(ft)["town"].unique()),
                               default=["Ang Mo Kio", "Punggol", "Queenstown", "Woodlands", "Tampines"])
        tp = T.town_prices(ft)
        sub = tp[tp["town"].isin(towns) & (tp["year"] >= 1995)]
        fig = px.line(sub, x="year", y="price_ref", color="town", markers=True,
                      labels={"price_ref": f"Constant-quality {ft} price ($)", "year": ""}, height=480)
        st.plotly_chart(fig, width="stretch")
        thin = sub[sub["n_town"] < 100]
        if not thin.empty:
            st.caption("Town-years with fewer than 100 sales are less precise: "
                       + ", ".join(f"{r.town} {r.year}" for r in thin.head(8).itertuples()))
        reg = store.table("hedonic_region_year.csv")
        fig = px.line(reg[reg["year"] >= 1995], x="year", y="price_ref_4room", color="region",
                      labels={"price_ref_4room": "Constant-quality 4-room price ($)", "year": ""},
                      title="By URA region", height=420)
        st.plotly_chart(fig, width="stretch")
        prem = tp[tp["year"] == tp["year"].max()].sort_values("location_premium_pct")
        fig = px.bar(prem, x="location_premium_pct", y="town", orientation="h", color="region",
                     labels={"location_premium_pct": "Location premium vs national (%)", "town": ""},
                     title=f"Location premium for an identical flat, {int(tp['year'].max())}", height=650)
        fig.update_yaxes(categoryorder="array", categoryarray=prem["town"].tolist())
        st.plotly_chart(fig, width="stretch")

    # ---- growth ----
    with tabs[2]:
        w = st.slider("Growth window (years)", 1, 10, 3)
        rg = T.rolling_growth_flags(window=w)
        rg = rg[rg["end_year"] >= 2000]
        piv = rg.pivot(index="town", columns="end_year", values="robust_z")
        fig = px.imshow(piv, color_continuous_scale="RdBu_r", zmin=-3, zmax=3, aspect="auto", height=700,
                        labels={"color": "robust z", "x": "end of window", "y": ""},
                        title=f"Trailing {w}-year constant-quality growth relative to other towns")
        st.plotly_chart(fig, width="stretch")
        flags = rg[rg["flag"] != ""].sort_values(["end_year", "robust_z"], ascending=[False, False])
        st.markdown(f"**{len(flags)} town-windows flagged** (|robust z| >= 2).")
        st.dataframe(flags.round(2), width="stretch", hide_index=True)

    # ---- lease ----
    with tabs[3]:
        st.markdown(
            "Is the market pricing lease decay more steeply than it used to? The hedonic "
            "regression is re-estimated each year, so the lease curve can change shape.")
        s = store.table("hedonic_fit_summary.csv")
        fig = px.line(s, x="year", y="lease_60_vs_90", markers=True,
                      labels={"lease_60_vs_90": "Price with 60 yrs left ÷ price with 90 yrs left", "year": ""},
                      title="The lease discount over time (all else equal)")
        st.plotly_chart(fig, width="stretch")
        lc = T.lease_curves()
        yrs = st.multiselect("Lease curves for years", sorted(lc["year"].unique()),
                             default=[2005, 2015, 2020, 2025])
        sub = lc[lc["year"].isin(yrs) & lc["in_support"]]
        fig = px.line(sub, x="remaining_lease", y="rel_price_vs_90", color=sub["year"].astype(str),
                      labels={"rel_price_vs_90": "Relative price (90 years = 1.0)",
                              "remaining_lease": "Remaining lease (years)", "color": "year"})
        st.plotly_chart(fig, width="stretch")
        st.caption("Curves are drawn only over the range of leases actually traded that year. "
                   "This is a market regularity, not a causal effect of lease alone: older flats also "
                   "differ in design, condition and estate.")
        if store.exists("lease_band_year.csv"):
            lb = T.lease_band_trends()
            fig = px.line(lb[lb["year"] >= 2000], x="year", y="median_psm", color="band",
                          labels={"median_psm": "Median price per sqm ($)", "year": "", "band": "Lease left"},
                          title="Raw price per sqm by remaining-lease band")
            st.plotly_chart(fig, width="stretch")

    # ---- raw ----
    with tabs[4]:
        df = load_hdb_data()
        c1, c2 = st.columns(2)
        towns = c1.multiselect("Town", get_town_list(), default=get_town_list()[:4], key="raw_t")
        fts = c2.multiselect("Flat type", get_flat_types(), default=["4-Room"], key="raw_f")
        sub = df[df["town"].isin(towns) & df["flat_type"].isin(fts) & (df["year_int"] >= 2010)]
        sub = sub.sort_values("quarter_sort")
        fig = px.line(sub, x="quarter", y="median_resale_price", color="town", line_group="flat_type",
                      symbol="flat_type", labels={"median_resale_price": "Median price ($)"}, height=460)
        st.plotly_chart(fig, width="stretch")
        st.caption("Raw quarterly medians: what actually sold, not adjusted for flat mix.")

    method_note(
        "Per year y: `log price = month + town + flat type + flat model + b1*lease + b2*lease^2 + "
        "b3*log(area) + b4*storey + b5*storey^2`, OLS on that year's transactions. The national "
        "index aggregates town effects with **fixed** 2015-2025 transaction-share weights, so it is "
        "not moved by shifts in where flats sell. Town effects are shared across flat types "
        "(no town x type interaction).")
