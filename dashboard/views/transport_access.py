"""Transport Access -- how rail access changed, and what happened to prices nearby."""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import store
from views._common import method_note, need


def show():
    st.title("Historical Transport Access")
    st.markdown(
        "Each flat is matched to the nearest MRT/LRT station that was **actually open in the month "
        "it was sold**, not today's network. v2 credited a 2010 sale with Downtown Line stations "
        "that opened in 2015-17; v3 does not.")
    if not need("rail_access_town_year.csv", "rail_event_study.csv"):
        return
    a = store.table("rail_access_town_year.csv")
    st_ = pd.read_csv(os.path.join(store.RAW, "rail_stations_opening_dates.csv"))

    st.header("How the network reached HDB estates")
    nat = a[a["town"] == "All HDB blocks"]
    c1, c2 = st.columns(2)
    fig = px.line(nat, x="year", y=["share_within_500m", "share_within_1km"], markers=True,
                  labels={"value": "% of existing HDB blocks", "year": "", "variable": ""},
                  title="Share of HDB blocks near an open station")
    c1.plotly_chart(fig, width="stretch")
    opened = st_.assign(year=st_["opened"].str[:4].astype(int)).groupby("year").size().reset_index(name="stations")
    fig = px.bar(opened, x="year", y="stations", title="Stations opened per year (MRT + LRT)",
                 labels={"stations": "Stations opened", "year": ""})
    c2.plotly_chart(fig, width="stretch")

    towns = sorted(t for t in a["town"].unique() if t != "All HDB blocks")
    sel = st.multiselect("Towns", towns, default=["Bukit Timah", "Bedok", "Hougang", "Yishun", "Marine Parade"])
    fig = px.line(a[a["town"].isin(sel)], x="year", y="share_within_1km", color="town", markers=True,
                  labels={"share_within_1km": "% of blocks within 1 km of an open station", "year": ""})
    st.plotly_chart(fig, width="stretch")
    y0, y1 = st.select_slider("Compare years", options=list(range(1990, 2027)), value=(2000, 2026))
    p = a[a["year"].isin([y0, y1]) & (a["town"] != "All HDB blocks")].pivot(
        index="town", columns="year", values="share_within_1km")
    if y0 in p and y1 in p:
        p = p.assign(change=p[y1] - p[y0]).dropna().sort_values("change").reset_index()
        fig = px.bar(p, x="change", y="town", orientation="h", height=650,
                     labels={"change": f"Change in % of blocks within 1 km, {y0}-{y1} (pp)", "town": ""})
        fig.update_yaxes(categoryorder="array", categoryarray=p["town"].tolist())
        st.plotly_chart(fig, width="stretch")
        st.caption("A town can lose ground when new estates are built far from existing stations "
                   "(only blocks that existed in a year are counted for that year).")

    st.header("What happened to prices when a station opened nearby?")
    es = store.table("rail_event_study.csv")
    fig = go.Figure()
    fig.add_scatter(x=es["k"], y=es["hi95_pct"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=es["k"], y=es["lo95_pct"], mode="lines", fill="tonexty", line=dict(width=0),
                    fillcolor="rgba(31,119,180,0.2)", name="95% interval")
    fig.add_scatter(x=es["k"], y=es["effect_pct"], mode="lines+markers", name="Estimate")
    fig.add_vline(x=-0.5, line_dash="dash", annotation_text="station opens")
    fig.add_hline(y=0, line_color="grey")
    fig.update_layout(height=430, xaxis_title="Years relative to opening",
                      yaxis_title="Price vs same-town comparison blocks (%), year -1 = 0")
    st.plotly_chart(fig, width="stretch")
    k = es.set_index("k")
    st.markdown(
        f"Blocks that first came within 1 km of an open station sold for about "
        f"**{k.loc[0, 'effect_pct']:+.1f}%** relative to comparison blocks in the opening year and "
        f"**{k.loc[6, 'effect_pct']:+.1f}%** six years later, compared with the year before. "
        f"But they were **already rising before opening** ({k.loc[-6, 'effect_pct']:+.1f}% six years "
        "before, relative to the year before opening).")
    st.warning(
        "**This is descriptive, not a causal estimate.** Stations are announced years before they open "
        "(prices can move on the announcement), and planners route lines to places that were changing "
        "anyway. The pre-opening upward trend means the 'parallel trends' assumption a causal "
        "difference-in-differences needs does not hold here. A credible causal estimate would need "
        "announcement dates and a design that handles them.")
    st.caption(f"{int(k['n_blocks'].median()):,} blocks per year on average; comparison = blocks in the same "
               "town and year more than 1 km from any open station and at least 6 years from getting one. "
               "Outcome = within-town price residual from that year's hedonic regression. 95% intervals "
               "from 300 bootstrap resamples of treated blocks.")

    with st.expander("Station opening dates used"):
        st.dataframe(st_, width="stretch", hide_index=True)
    method_note(
        "**Distances** are straight-line from each block (OneMap coordinates) to the station's exit "
        "centroid (LTA). **Opening dates** come from Wikipedia's lists of MRT and LRT stations and the "
        "stations' own articles; LRT dates were hand-checked because Wikipedia's LRT table shares one "
        "date per loop. Interchanges take the earliest opening at that location. Ten Mile Junction LRT "
        "(1999-2019) is missing from the 2025 LTA file and is omitted.\n\n"
        "**Used in the price model**: the served model's distance-to-rail feature is now this "
        "time-correct distance.")
